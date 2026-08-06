//! Alternate-screen dashboard for `--tui`.
//!
//! Rendering is intentionally downstream of `ui`'s structured line tap and
//! `System::snapshot()`: the dashboard observes the harness, but it is not a
//! second logging or command implementation.

use crate::api::{Confidence, Spend, Usage};
use crate::system::{Status, System, SystemSnapshot};
use crate::ui;
use anyhow::Context;
use crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event as TerminalEvent, EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEvent,
    MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use futures_util::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Borders, List, ListItem, ListState, Padding, Paragraph};
use ratatui::{Frame, Terminal};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};
use tokio::sync::broadcast;

#[cfg(unix)]
use std::fs::File;
#[cfg(unix)]
use std::io::Write;
#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd, RawFd};
#[cfg(unix)]
use std::os::unix::net::UnixStream;

const FEED_CAP: usize = 1_500;
const SNAPSHOT_TICK: Duration = Duration::from_millis(250);
const REDRAW_DEBOUNCE: Duration = Duration::from_millis(33);
const WARN_MOOD: Duration = Duration::from_secs(5);
const WHEEL_SCROLL_LINES: usize = 3;
/// Braille spinner frames and how long each one is held. 120ms reads as motion
/// without turning the transcript into a strobe, and the frames are plain
/// UTF-8 with no dependency behind them.
const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const SPINNER_PERIOD: Duration = Duration::from_millis(120);
/// Rendered rows one event body keeps in the summarized view, and the slack
/// a body may run over that before clamping is worth it: swapping two rows for
/// a marker row saves one row and costs the reader the content, so only clamp
/// once the marker buys back at least two rows.
const CLAMP_ROWS: usize = 8;
const CLAMP_SLACK: usize = 2;

struct DashboardState {
    snapshot: SystemSnapshot,
    feed: VecDeque<ui::Event>,
    selected: Option<String>,
    show_traces: bool,
    input: String,
    /// Rendered transcript lines above the newest visible page.
    scroll_from_bottom: usize,
    max_scroll: usize,
    activity_page: usize,
    last_warn: Option<Instant>,
    animation_frame: bool,
    /// When each currently-working process started its turn. The snapshot
    /// carries no timestamps, so the dashboard remembers this itself.
    working_since: HashMap<String, Instant>,
    /// Fixed origin for the spinner phase, so frames advance with wall time
    /// rather than with however often the terminal happened to redraw.
    started: Instant,
    /// Whether the terminal is reporting the mouse to us. While it is, the
    /// terminal hands us drags instead of doing its own click-and-drag
    /// selection, so copying text out of the transcript is impossible; the
    /// human can hand the mouse back to the terminal with ctrl-o.
    mouse_capture: bool,
}

impl DashboardState {
    fn new(snapshot: SystemSnapshot) -> Self {
        Self {
            snapshot,
            feed: VecDeque::new(),
            selected: None,
            show_traces: false,
            input: String::new(),
            scroll_from_bottom: 0,
            max_scroll: 0,
            activity_page: 1,
            last_warn: None,
            animation_frame: false,
            working_since: HashMap::new(),
            started: Instant::now(),
            mouse_capture: true,
        }
    }

    fn push(&mut self, mut event: ui::Event) {
        if event.kind == "warn" {
            self.last_warn = Some(Instant::now());
        }
        // Sanitize once at the trust boundary rather than on every redraw.
        event.who = safe_inline(&event.who);
        event.process = event.process.map(|process| safe_inline(&process));
        event.text = terminal_safe(&event.text);
        // Providers stream assistant text one line at a time. Keeping those as
        // separate transcript entries repeats the speaker label down the
        // screen; adjacent lines from the same response are one visual turn.
        if event.kind == "say"
            && let Some(previous) = self.feed.back_mut().filter(|previous| {
                previous.kind == "say"
                    && previous.who == event.who
                    && previous.process == event.process
            })
        {
            previous.text.push('\n');
            previous.text.push_str(&event.text);
            return;
        }
        self.feed.push_back(event);
        while self.feed.len() > FEED_CAP {
            self.feed.pop_front();
        }
    }

    fn refresh(&mut self, snapshot: SystemSnapshot) {
        if self
            .selected
            .as_ref()
            .is_some_and(|id| !snapshot.processes.iter().any(|process| &process.id == id))
        {
            self.selected = None;
        }
        // A spinner has to say how long *this* turn has been going, and a
        // snapshot has no timestamps. Remember when each process was first seen
        // working and forget it the moment it stops, so the next turn starts
        // from zero rather than from when the process was spawned.
        let now = Instant::now();
        self.working_since.retain(|id, _| {
            snapshot
                .processes
                .iter()
                .any(|process| &process.id == id && process.status == Status::Running)
        });
        for process in &snapshot.processes {
            if process.status == Status::Running {
                self.working_since.entry(process.id.clone()).or_insert(now);
            }
        }
        self.snapshot = snapshot;
        self.animation_frame = !self.animation_frame;
    }

    /// How long the named process has been working, if it is.
    fn working_for(&self, id: &str) -> Option<Duration> {
        self.working_since.get(id).map(|since| since.elapsed())
    }

    /// The spinner as it stands right now: one frame per `SPINNER_PERIOD` since
    /// the dashboard opened, so every spinner on screen turns in step.
    fn spinner(&self) -> &'static str {
        spinner_at(self.started.elapsed())
    }

    fn warning_recent(&self) -> bool {
        self.last_warn
            .is_some_and(|when| when.elapsed() <= WARN_MOOD)
    }

    fn scroll_older(&mut self, lines: usize) {
        self.scroll_from_bottom = self
            .scroll_from_bottom
            .saturating_add(lines)
            .min(self.max_scroll);
    }

    fn scroll_newer(&mut self, lines: usize) {
        self.scroll_from_bottom = self.scroll_from_bottom.saturating_sub(lines);
    }

    fn latest(&mut self) {
        self.scroll_from_bottom = 0;
    }
}

#[cfg(unix)]
type DashboardOutput = File;
#[cfg(not(unix))]
type DashboardOutput = std::io::Stdout;

/// Restores the caller's terminal even when rendering or input fails. On
/// Unix, ratatui writes through a duplicate of the original terminal while
/// stdout and stderr are redirected to a local capture socket. This keeps a
/// dependency's stray `println!`, panic hook, or native log from corrupting
/// ratatui's idea of what is currently on screen.
struct TerminalGuard {
    #[cfg(unix)]
    control: File,
    #[cfg(unix)]
    saved_stdout: File,
    #[cfg(unix)]
    saved_stderr: File,
    #[cfg(unix)]
    capture_thread: Option<std::thread::JoinHandle<()>>,
}

impl TerminalGuard {
    #[cfg(unix)]
    fn enter() -> anyhow::Result<(Self, DashboardOutput)> {
        let saved_stdout =
            duplicate_fd(libc::STDOUT_FILENO).context("could not preserve terminal stdout")?;
        let saved_stderr =
            duplicate_fd(libc::STDERR_FILENO).context("could not preserve terminal stderr")?;
        let mut control = saved_stdout
            .try_clone()
            .context("could not duplicate the terminal control handle")?;
        let renderer = saved_stdout
            .try_clone()
            .context("could not duplicate the terminal render handle")?;
        let (capture_reader, capture_writer) =
            UnixStream::pair().context("could not create the TUI output capture")?;

        // Flush anything queued for the normal console before fd 1 changes
        // meaning underneath Rust's global stdout handle.
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        enable_raw_mode().context("could not enable terminal raw mode")?;
        if let Err(error) = execute!(
            control,
            EnterAlternateScreen,
            EnableBracketedPaste,
            EnableMouseCapture
        ) {
            let _ = execute!(
                control,
                DisableMouseCapture,
                DisableBracketedPaste,
                LeaveAlternateScreen
            );
            let _ = disable_raw_mode();
            return Err(error).context("could not enter the alternate screen");
        }

        if let Err(error) = redirect_fd(&capture_writer, libc::STDOUT_FILENO)
            .and_then(|()| redirect_fd(&capture_writer, libc::STDERR_FILENO))
        {
            let _ = restore_fd(&saved_stdout, libc::STDOUT_FILENO);
            let _ = restore_fd(&saved_stderr, libc::STDERR_FILENO);
            let _ = execute!(
                control,
                DisableMouseCapture,
                DisableBracketedPaste,
                LeaveAlternateScreen
            );
            let _ = disable_raw_mode();
            return Err(error).context("could not quarantine terminal output");
        }
        drop(capture_writer);

        let capture_thread = match std::thread::Builder::new()
            .name("bitty-tui-output".into())
            .spawn(move || capture_external_output(capture_reader))
        {
            Ok(thread) => thread,
            Err(error) => {
                let _ = restore_fd(&saved_stdout, libc::STDOUT_FILENO);
                let _ = restore_fd(&saved_stderr, libc::STDERR_FILENO);
                let _ = execute!(
                    control,
                    DisableMouseCapture,
                    DisableBracketedPaste,
                    LeaveAlternateScreen
                );
                let _ = disable_raw_mode();
                return Err(error).context("could not start the TUI output capture");
            }
        };

        Ok((
            Self {
                control,
                saved_stdout,
                saved_stderr,
                capture_thread: Some(capture_thread),
            },
            renderer,
        ))
    }

    /// Hand the mouse back to the terminal, or take it again, without leaving
    /// the alternate screen. Written to the saved terminal handle rather than
    /// to stdout, which by this point is a pipe into the transcript.
    #[cfg(unix)]
    fn set_mouse_capture(&mut self, capture: bool) -> std::io::Result<()> {
        if capture {
            execute!(self.control, EnableMouseCapture)
        } else {
            execute!(self.control, DisableMouseCapture)
        }
    }

    #[cfg(not(unix))]
    fn set_mouse_capture(&mut self, capture: bool) -> std::io::Result<()> {
        if capture {
            execute!(std::io::stdout(), EnableMouseCapture)
        } else {
            execute!(std::io::stdout(), DisableMouseCapture)
        }
    }

    #[cfg(not(unix))]
    fn enter() -> anyhow::Result<(Self, DashboardOutput)> {
        enable_raw_mode().context("could not enable terminal raw mode")?;
        if let Err(error) = execute!(
            std::io::stdout(),
            EnterAlternateScreen,
            EnableBracketedPaste,
            EnableMouseCapture
        ) {
            let _ = execute!(
                std::io::stdout(),
                DisableMouseCapture,
                DisableBracketedPaste,
                LeaveAlternateScreen
            );
            let _ = disable_raw_mode();
            return Err(error).context("could not enter the alternate screen");
        }
        Ok((Self {}, std::io::stdout()))
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            // The renderer has already been dropped. Leave through the saved
            // terminal descriptor, then restore the process-wide descriptors
            // before waiting for the capture reader to observe EOF.
            // Mouse capture is disabled unconditionally even if ctrl-o already
            // turned it off: the sequence is idempotent, and teardown that
            // trusts a flag is teardown that can leave a terminal reporting
            // the mouse forever after a panic.
            let _ = execute!(
                self.control,
                DisableMouseCapture,
                DisableBracketedPaste,
                LeaveAlternateScreen
            );
            let _ = disable_raw_mode();
            let _ = std::io::stdout().flush();
            let _ = std::io::stderr().flush();
            let _ = restore_fd(&self.saved_stdout, libc::STDOUT_FILENO);
            let _ = restore_fd(&self.saved_stderr, libc::STDERR_FILENO);
            if let Some(thread) = self.capture_thread.take() {
                let _ = thread.join();
            }
        }

        #[cfg(not(unix))]
        {
            let _ = disable_raw_mode();
            let _ = execute!(
                std::io::stdout(),
                DisableMouseCapture,
                DisableBracketedPaste,
                LeaveAlternateScreen
            );
        }
    }
}

#[cfg(unix)]
fn duplicate_fd(fd: RawFd) -> std::io::Result<File> {
    // SAFETY: `dup` either returns a new owned descriptor or -1. Converting a
    // successful result to `File` transfers exactly that new ownership.
    let duplicate = unsafe { libc::dup(fd) };
    if duplicate < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        // SAFETY: the descriptor came from a successful `dup` above and is not
        // owned anywhere else.
        Ok(unsafe { File::from_raw_fd(duplicate) })
    }
}

#[cfg(unix)]
fn redirect_fd(source: &impl AsRawFd, target: RawFd) -> std::io::Result<()> {
    // SAFETY: both descriptors are valid for this call; `dup2` atomically
    // replaces `target` without taking ownership of `source`.
    if unsafe { libc::dup2(source.as_raw_fd(), target) } < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(unix)]
fn restore_fd(source: &File, target: RawFd) -> std::io::Result<()> {
    redirect_fd(source, target)
}

#[cfg(unix)]
fn capture_external_output(mut reader: UnixStream) {
    const CHUNK: usize = 8 * 1024;
    let mut read_buffer = [0_u8; CHUNK];
    let mut pending = Vec::new();

    loop {
        let count = match std::io::Read::read(&mut reader, &mut read_buffer) {
            Ok(0) | Err(_) => break,
            Ok(count) => count,
        };
        for byte in &read_buffer[..count] {
            if *byte == b'\n' {
                publish_external(&pending);
                pending.clear();
            } else {
                pending.push(*byte);
                // A writer is not allowed to make the capture thread retain
                // an unbounded line. Chunk it without blocking that writer.
                if pending.len() == CHUNK {
                    publish_external(&pending);
                    pending.clear();
                }
            }
        }
    }
    publish_external(&pending);
}

#[cfg(unix)]
fn publish_external(bytes: &[u8]) {
    if !bytes.is_empty() {
        ui::external(&String::from_utf8_lossy(bytes));
    }
}

/// Run the dashboard until the user dispatches `/quit` (or presses Ctrl-C).
/// The callback returns true when its command means quit; every other command
/// is handled by the same dispatcher plain mode uses.
pub async fn run<F>(
    system: &System,
    session: &str,
    mut events: broadcast::Receiver<ui::Event>,
    mut dispatch: F,
) -> anyhow::Result<()>
where
    F: FnMut(&str) -> bool,
{
    let (mut guard, output) = TerminalGuard::enter()?;
    let backend = CrosstermBackend::new(output);
    let mut terminal = Terminal::new(backend).context("could not initialize the dashboard")?;

    let mut state = DashboardState::new(system.snapshot());
    let mut input_events = EventStream::new();
    let mut snapshot_tick = tokio::time::interval(SNAPSHOT_TICK);
    let mut redraw_tick = tokio::time::interval(REDRAW_DEBOUNCE);
    let mut spinner_tick = tokio::time::interval(SPINNER_PERIOD);
    snapshot_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    redraw_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    spinner_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut tap_open = true;
    let mut dirty = true;
    // What the terminal is actually doing, as opposed to what the human has
    // asked for in `state.mouse_capture`.
    let mut capturing = true;

    loop {
        tokio::select! {
            terminal_event = input_events.next() => {
                match terminal_event {
                    Some(Ok(TerminalEvent::Key(key))) => {
                        if handle_key(key, &mut state, &mut dispatch) {
                            break;
                        }
                        if state.mouse_capture != capturing {
                            match guard.set_mouse_capture(state.mouse_capture) {
                                Ok(()) => capturing = state.mouse_capture,
                                // Never let the indicator claim a mode the
                                // terminal refused to enter.
                                Err(_) => state.mouse_capture = capturing,
                            }
                        }
                        dirty = true;
                    }
                    Some(Ok(TerminalEvent::Paste(text))) => {
                        // The command line is deliberately one line. Preserve
                        // pasted words while preventing embedded newlines from
                        // dispatching surprise commands.
                        for character in text.chars() {
                            state.input.push(if character.is_control() { ' ' } else { character });
                        }
                        dirty = true;
                    }
                    Some(Ok(TerminalEvent::Mouse(mouse))) => {
                        handle_mouse(mouse, &mut state);
                        dirty = true;
                    }
                    Some(Ok(TerminalEvent::Resize(_, _))) => dirty = true,
                    Some(Ok(_)) => {}
                    Some(Err(error)) => return Err(error).context("terminal input failed"),
                    None => break,
                }
            }
            tapped = events.recv(), if tap_open => {
                match tapped {
                    Ok(event) => {
                        state.push(event);
                        dirty = true;
                    }
                    // A busy system is allowed to outrun its observer. Resume
                    // at the newest available event rather than blocking the
                    // actors that produced it.
                    Err(broadcast::error::RecvError::Lagged(_)) => {}
                    Err(broadcast::error::RecvError::Closed) => tap_open = false,
                }
            }
            _ = snapshot_tick.tick() => {
                state.refresh(system.snapshot());
                dirty = true;
            }
            // Only while something is actually working. With every process
            // settled this arm is disabled, so an idle dashboard waits on
            // events exactly as it did before instead of animating nothing.
            _ = spinner_tick.tick(), if !state.snapshot.settled => {
                dirty = true;
            }
            _ = redraw_tick.tick(), if dirty => {
                terminal
                    .draw(|frame| render(frame, &mut state, session))
                    .context("dashboard redraw failed")?;
                dirty = false;
            }
        }
    }

    Ok(())
}

fn handle_key<F>(key: KeyEvent, state: &mut DashboardState, dispatch: &mut F) -> bool
where
    F: FnMut(&str) -> bool,
{
    if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
        return false;
    }
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        match key.code {
            KeyCode::Char('c') => return true,
            KeyCode::Char('t') => {
                state.show_traces = !state.show_traces;
                state.latest();
            }
            // Ctrl-O rather than ctrl-m, which a terminal delivers as Enter,
            // and rather than a bare letter, which belongs to the composer.
            KeyCode::Char('o') => state.mouse_capture = !state.mouse_capture,
            _ => {}
        }
        return false;
    }

    match key.code {
        KeyCode::Enter => {
            let line = std::mem::take(&mut state.input);
            state.latest();
            if !line.trim().is_empty() && dispatch(&line) {
                return true;
            }
        }
        KeyCode::Backspace | KeyCode::Delete => {
            state.input.pop();
        }
        KeyCode::Esc => {
            state.selected = None;
            state.latest();
        }
        KeyCode::Up => move_selection(state, -1),
        KeyCode::Down => move_selection(state, 1),
        KeyCode::PageUp => state.scroll_older(state.activity_page),
        KeyCode::PageDown => state.scroll_newer(state.activity_page),
        KeyCode::Home => state.scroll_older(state.max_scroll),
        KeyCode::End => state.latest(),
        KeyCode::Char(character)
            if !key
                .modifiers
                .intersects(KeyModifiers::ALT | KeyModifiers::SUPER) =>
        {
            state.input.push(character);
        }
        _ => {}
    }
    false
}

fn handle_mouse(mouse: MouseEvent, state: &mut DashboardState) {
    match mouse.kind {
        MouseEventKind::ScrollUp => state.scroll_older(WHEEL_SCROLL_LINES),
        MouseEventKind::ScrollDown => state.scroll_newer(WHEEL_SCROLL_LINES),
        _ => {}
    }
}

fn move_selection(state: &mut DashboardState, delta: isize) {
    let rows = tree_rows(&state.snapshot);
    if rows.is_empty() {
        state.selected = None;
        return;
    }
    let current = state.selected.as_ref().and_then(|id| {
        rows.iter()
            .position(|row| state.snapshot.processes[row.index].id == *id)
    });
    let next = match (current, delta.is_negative()) {
        (Some(index), true) => index.saturating_sub(1),
        (Some(index), false) => (index + 1).min(rows.len() - 1),
        (None, true) => rows.len() - 1,
        (None, false) => 0,
    };
    state.selected = Some(state.snapshot.processes[rows[next].index].id.clone());
    state.latest();
}

fn render(frame: &mut Frame<'_>, state: &mut DashboardState, session: &str) {
    let area = frame.area();
    if area.width < 56 || area.height < 12 {
        frame.render_widget(
            Paragraph::new("bitty TUI needs a terminal at least 56×12")
                .style(Style::default().fg(Color::Yellow))
                .block(Block::default().borders(Borders::ALL).title(" bitty ")),
            area,
        );
        return;
    }

    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(4),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(area);
    render_header(frame, vertical[0], state, session);

    // The transcript owns the screen; processes are a slim navigation rail.
    let tree_width = (area.width / 4).clamp(24, 34);
    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(tree_width), Constraint::Min(20)])
        .split(vertical[1]);
    render_processes(frame, body[0], state);
    render_activity(frame, body[1], state);
    render_input(frame, vertical[2], state);
    render_status_line(frame, vertical[3], state, session);
}

fn render_header(frame: &mut Frame<'_>, area: Rect, state: &DashboardState, session: &str) {
    let warning = state.warning_recent();
    let face = if warning {
        "(⊙ω⊙)!"
    } else if state.snapshot.settled {
        "(-ω-)ᶻᶻ"
    } else if state.animation_frame {
        "(^･ω･^)~"
    } else {
        "~(^･ω･^)"
    };
    let cat_style = if warning {
        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
    } else if state.snapshot.settled {
        Style::default().fg(Color::Blue)
    } else {
        Style::default().fg(Color::Cyan)
    };
    let (state_word, state_style, state_glyph) = if state.snapshot.settled {
        ("idle", Style::default().fg(Color::Blue), "○")
    } else {
        ("working", Style::default().fg(Color::Green), "●")
    };
    let lines = vec![
        Line::from(vec![
            Span::styled(format!(" {face:<11}"), cat_style),
            Span::styled("bitty", Style::default().add_modifier(Modifier::BOLD)),
            Span::styled(
                format!("  {}", safe_inline(session)),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(vec![
            Span::raw("             "),
            Span::styled(format!("{state_glyph} "), state_style),
            Span::styled(state_word, state_style.add_modifier(Modifier::BOLD)),
            Span::styled(
                system_summary(&state.snapshot),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
    ];
    frame.render_widget(
        Paragraph::new(Text::from(lines)).block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(Color::DarkGray)),
        ),
        area,
    );
}

/// The run's own numbers. They belong beside the cat rather than on the status
/// line, which now speaks for one process at a time — a whole-system peak sat
/// there for a long time pretending to be the selected process's context.
/// Zero-valued figures are omitted: a run that has billed nothing has nothing
/// to say about it, and the header is only as wide as the terminal.
fn system_summary(snapshot: &SystemSnapshot) -> String {
    let mut summary = format!("  ·  {}", process_count(snapshot.processes.len()));
    // Where the run's tokens actually went: cache reads cost around a tenth of
    // fresh input, so the share is the difference between a cheap run and an
    // expensive one. "cached" on its own was read as how much of the prompt is
    // sitting in the cache, which is a different and less useful number, and
    // the figure covers the whole run rather than the process the rest of the
    // frame is about — both of which the label now says out loud.
    if let Some(share) = cache_share(snapshot.spend.usage) {
        summary.push_str(&format!("  ·  run {share} cache hits"));
    }
    if snapshot.billable > 0 {
        summary.push_str(&format!(
            "  ·  {} billable",
            format_tokens(snapshot.billable)
        ));
    }
    if snapshot.peak_context > 0 {
        summary.push_str(&format!(
            "  ·  peak {} ctx",
            format_tokens(snapshot.peak_context)
        ));
    }
    summary
}

fn render_processes(frame: &mut Frame<'_>, area: Rect, state: &DashboardState) {
    let rows = tree_rows(&state.snapshot);
    let chosen = state.selected.as_ref().and_then(|id| {
        rows.iter()
            .position(|row| state.snapshot.processes[row.index].id == *id)
    });
    // One right border, a column of padding either side, and the two columns
    // the selection marker holds open on every row. The rail clips rather than
    // wraps, so the detail row is fitted to what is left by hand.
    let detail_width = area.width.saturating_sub(5).max(1) as usize;
    let items: Vec<ListItem<'_>> = rows
        .iter()
        .enumerate()
        .map(|(position, row)| {
            let process = &state.snapshot.processes[row.index];
            let selected = chosen == Some(position);
            let (glyph, glyph_style) = status_glyph(process.status);
            // A working process turns; an idle or stopped one keeps its glyph.
            let working = state.working_for(&process.id);
            let glyph = if working.is_some() {
                state.spinner()
            } else {
                glyph
            };
            let name = process
                .name
                .as_ref()
                .map(|name| format!(" {}", safe_inline(name)))
                .unwrap_or_default();
            let indent = row.prefix.chars().count();
            // The timer only earns its place if it fits: the rail is narrow and
            // a clipped id is worse than a missing stopwatch.
            let head = indent + 2 + process.id.chars().count() + name.chars().count();
            let elapsed = working
                .map(|elapsed| format!("  {}", format_elapsed(elapsed)))
                .filter(|label| head + label.chars().count() <= detail_width)
                .unwrap_or_default();
            // Selection is a marker and a band, never a repaint: every colour
            // on the row means something already — whose process it is, what it
            // is doing, how big its model — and a cyan wash erased all three.
            let marker = if selected {
                Span::styled(
                    "› ",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )
            } else {
                Span::raw("  ")
            };
            // Dim text has to lift off the band to stay readable.
            let quiet = Style::default().fg(if selected {
                Color::Gray
            } else {
                Color::DarkGray
            });
            let rule = Style::default().fg(if selected { Color::DarkGray } else { TREE_RULE });
            let (chip, chip_color) = size_chip(&safe_inline(&process.runs));
            ListItem::new(Text::from(vec![
                Line::from(vec![
                    marker,
                    Span::styled(row.prefix.clone(), rule),
                    Span::styled(format!("{glyph} "), glyph_style),
                    Span::styled(
                        process.id.clone(),
                        Style::default()
                            .fg(process_color(&process.id))
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(name),
                    Span::styled(elapsed, quiet),
                ]),
                Line::from(vec![
                    Span::raw(" ".repeat(2 + indent)),
                    Span::styled(chip, Style::default().fg(chip_color)),
                    Span::raw(" "),
                    Span::styled(
                        process_detail(
                            process.tokens,
                            process.spend,
                            detail_width.saturating_sub(indent + 2),
                        ),
                        quiet,
                    ),
                ]),
            ]))
        })
        .collect();

    let mut list_state = ListState::default();
    list_state.select(chosen);
    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::RIGHT)
                // The rail's edge is the strongest divider in the frame, so it
                // is the only heavy rule: inside the rail the tree's pipes are
                // thin and dim, and the fields on a row are parted by a dot.
                .border_type(BorderType::Thick)
                .border_style(Style::default().fg(Color::DarkGray))
                .padding(Padding::new(1, 1, 1, 0))
                .title(Span::styled(
                    " processes ",
                    Style::default().fg(Color::DarkGray),
                )),
        )
        .highlight_style(Style::default().bg(SELECTED_BG));
    frame.render_stateful_widget(list, area, &mut list_state);
}

fn render_activity(frame: &mut Frame<'_>, area: Rect, state: &mut DashboardState) {
    let block = Block::default().padding(Padding::new(2, 2, 1, 0));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let mut lines = activity_lines(state, inner.width.max(1) as usize);
    if lines.is_empty() {
        lines.push(Line::styled(
            "Waiting for activity…",
            Style::default().fg(Color::DarkGray),
        ));
    }
    let visible = (inner.height as usize).max(1);
    let (window, max_scroll, scroll) =
        activity_window(lines.len(), visible, state.scroll_from_bottom);
    state.activity_page = visible;
    state.max_scroll = max_scroll;
    state.scroll_from_bottom = scroll;
    frame.render_widget(Paragraph::new(Text::from(lines[window].to_vec())), inner);
}

fn render_input(frame: &mut Frame<'_>, area: Rect, state: &DashboardState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = block.inner(area);
    // Reserve the last cell for the cursor when the line fills the field.
    let width = inner.width.saturating_sub(3) as usize;
    let visible: String = state
        .input
        .chars()
        .rev()
        .take(width)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let cursor = visible.chars().count() as u16 + 2;
    frame.render_widget(block, area);
    let input = if visible.is_empty() {
        Line::from(vec![
            Span::styled("› ", Style::default().fg(Color::Cyan)),
            Span::styled(
                "Send a message or /command",
                Style::default().fg(Color::DarkGray),
            ),
        ])
    } else {
        Line::from(vec![
            Span::styled("› ", Style::default().fg(Color::Cyan)),
            Span::raw(visible),
        ])
    };
    frame.render_widget(Paragraph::new(input), inner);
    if inner.width > 0 {
        frame.set_cursor_position((inner.x.saturating_add(cursor), inner.y));
    }
}

fn render_status_line(frame: &mut Frame<'_>, area: Rect, state: &DashboardState, session: &str) {
    let active = state
        .selected
        .as_ref()
        .and_then(|id| {
            state
                .snapshot
                .processes
                .iter()
                .find(|process| &process.id == id)
        })
        .or_else(|| state.snapshot.processes.first());
    let runs = active
        .map(|process| safe_inline(&process.runs))
        .unwrap_or_else(|| "no model".into());
    // The old line showed the whole system's peak context here and called it
    // "context", which says nothing about the process being looked at. This is
    // that process's live context instead; the peak is gone rather than shown
    // twice under two labels, since the rail now carries every process's own.
    // Working state for the process this line speaks for, so the human can see
    // that something is happening without reading the transcript.
    let turn = active
        .and_then(|process| state.working_for(&process.id))
        .map(|elapsed| format!(" {} {}", state.spinner(), format_elapsed(elapsed)))
        .unwrap_or_default();
    let context = format_tokens(active.map_or(0, |process| process.tokens));
    let cost = format_cost(active.map(|process| process.spend).unwrap_or_default());
    let run = format_cost(state.snapshot.spend);
    let traces = if state.show_traces {
        "traces on"
    } else {
        "traces summarized"
    };
    let scroll = if state.scroll_from_bottom == 0 {
        String::new()
    } else {
        format!(" · {} lines back · End latest", state.scroll_from_bottom)
    };
    let dim = Style::default().fg(Color::DarkGray);
    let mut spans = vec![Span::styled(
        // Billable tokens dropped from this line: cost answers the same
        // question in the unit the human asked about, and the line is full.
        format!(" {runs}{turn} · {context} ctx · cost {cost} · run {run}"),
        dim,
    )];
    // Only announced while it is off: a dead scroll wheel is the surprise that
    // needs explaining, and the way back is worth more than the label. It sits
    // ahead of the session name because the tail of this line is the first
    // thing a narrow terminal clips. Cyan, which already means "you" here: it
    // is a mode the human chose, not a warning.
    if !state.mouse_capture {
        spans.push(Span::styled(" · mouse off · ctrl-o", dim.fg(Color::Cyan)));
    }
    spans.push(Span::styled(format!(" · {}", safe_inline(session)), dim));
    // The filter names one process, so it wears that process's colour: the
    // status line, the process list and the transcript then agree about who.
    if let Some(id) = state.selected.as_deref() {
        spans.push(Span::styled(
            format!(" · filter {id}"),
            dim.fg(process_color(id)),
        ));
    }
    spans.push(Span::styled(
        format!("{scroll} · {traces} · ↑↓ agents · esc clear · ctrl-t traces · ctrl-o mouse"),
        dim,
    ));
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn activity_lines(state: &DashboardState, width: usize) -> Vec<Line<'static>> {
    let visible: Vec<&ui::Event> = state
        .feed
        .iter()
        .filter(|event| {
            state
                .selected
                .as_deref()
                .is_none_or(|id| event_belongs_to(event, id))
        })
        .collect();
    let mut lines = Vec::new();
    let mut index = 0;
    while index < visible.len() {
        let event = visible[index];
        // Hidden traces are summarized rather than dropped: with them gone
        // entirely the transcript says nothing about what an agent is doing
        // between the messages it sends.
        if !state.show_traces && event.kind == "trace" {
            let end = trace_run_end(&visible, index);
            let steps: Vec<&str> = visible[index..end]
                .iter()
                .map(|step| step.text.as_str())
                .collect();
            let prefix = trace_prefix(&event.who);
            let summary = format!("{prefix}{}", collapsed_trace_text(&steps));
            lines.push(labelled_line(
                elide_to_width(&summary, width),
                prefix.chars().count(),
                speaker_color(event),
                event_style(event.kind),
            ));
            index = end;
            // Set the machinery off from the messages around it. A stretch of
            // consecutive collapsed runs is one block, so the separator lands
            // after the last of them: blank-separating every summary would
            // spend the rows the collapse just saved and read as sparse.
            if visible.get(end).is_none_or(|next| next.kind != "trace") {
                lines.push(Line::default());
            }
            continue;
        }
        let who = &event.who;
        let recipient = event.process.as_deref().unwrap_or("?");
        let selected_recipient = state.selected.as_deref() == event.process.as_deref();
        let prefix = match event.kind {
            "user" => "› You  ".to_string(),
            "incoming" if event.who == "user" && selected_recipient => "› You  ".to_string(),
            "incoming" if event.who == "user" => format!("› You → {recipient}  "),
            "incoming" if selected_recipient => format!("← {who}  "),
            "incoming" => format!("← {who} → {recipient}  "),
            "say" if !event.who.is_empty() => format!("• {who}  "),
            "trace" => trace_prefix(who),
            "mail" if !event.who.is_empty() => format!("› {who} → you  "),
            "warn" if !event.who.is_empty() => format!("! {who}  "),
            "external" => "  ↳ external output  ".to_string(),
            "system" => "  ".to_string(),
            _ if !event.who.is_empty() => format!("• {who}  "),
            _ => "  ".to_string(),
        };
        let style = event_style(event.kind);
        let mut rows = Vec::new();
        for (index, source_line) in event.text.split('\n').enumerate() {
            let lead = if index == 0 {
                prefix.as_str()
            } else if prefix.is_empty() {
                ""
            } else {
                "  "
            };
            rows.extend(wrap_chars(&format!("{lead}{source_line}"), width));
        }
        // Same bargain as a collapsed trace run: in the summarized view a
        // message worth pages of rows buries everything around it. Clamp the
        // bodies that arrive from elsewhere and say what ctrl-t would show;
        // never clamp what the human typed, and never clamp a warning.
        let full = rows.len();
        if !state.show_traces && matches!(event.kind, "mail" | "incoming" | "say" | "external") {
            rows = clamp_rows(rows, CLAMP_ROWS, width);
        }
        // The clamp only ever appends, so a changed row count means the last
        // row is the marker.
        let marker = (rows.len() != full).then(|| rows.len() - 1);
        let label = speaker_color(event);
        let label_columns = prefix.chars().count();
        for (row, text) in rows.into_iter().enumerate() {
            // The marker is furniture, not content, so it reads as dim as a
            // collapsed trace run rather than as more of the message.
            let row_style = if Some(row) == marker {
                Style::default().fg(Color::DarkGray)
            } else {
                style
            };
            // Only the first row carries the speaker label; continuation rows
            // are all body.
            if row == 0 {
                lines.push(labelled_line(text, label_columns, label, row_style));
            } else {
                lines.push(Line::styled(text, row_style));
            }
        }
        if matches!(event.kind, "user" | "incoming" | "say" | "mail" | "warn") {
            lines.push(Line::default());
        }
        index += 1;
    }
    // A separator belongs between things. At the bottom of a bottom-aligned
    // transcript it is just the newest row spent on nothing, so the rule is
    // the same for every kind: separate, then drop the trailing one.
    while lines.last().is_some_and(|line| line.spans.is_empty()) {
        lines.pop();
    }
    lines
}

/// The end of the maximal run of consecutive traces from one speaker. Runs
/// are per speaker so two agents tracing at the same time stay apart.
fn trace_run_end(events: &[&ui::Event], start: usize) -> usize {
    let first = events[start];
    events[start..]
        .iter()
        .position(|event| {
            event.kind != "trace" || event.who != first.who || event.process != first.process
        })
        .map_or(events.len(), |offset| start + offset)
}

fn trace_prefix(who: &str) -> String {
    if who.is_empty() {
        "  ".to_string()
    } else {
        format!("  ↳ {who}  ")
    }
}

/// The body of a collapsed run: how much was folded away, plus the one step
/// worth reading. A failed step is what the human most needs to see; short of
/// that the newest step is the one that says what the agent is doing now.
fn collapsed_trace_text(steps: &[&str]) -> String {
    let highlight = steps
        .iter()
        .rev()
        .find(|step| step.contains('✗'))
        .or_else(|| steps.last())
        .copied()
        .unwrap_or_default();
    let mut source = highlight.split('\n');
    let mut headline = source.next().unwrap_or_default().trim().to_string();
    if source.next().is_some() {
        headline.push('…');
    }
    if steps.len() > 1 {
        format!("… {} steps · {headline}", steps.len())
    } else {
        headline
    }
}

/// The rows of one event body to actually render. Pure: it takes rows that
/// are already wrapped, so a single 4000-character paragraph clamps exactly
/// like a 40-line one, and the caller keeps every other rendering decision.
fn clamp_rows(rows: Vec<String>, limit: usize, width: usize) -> Vec<String> {
    if rows.len() <= limit + CLAMP_SLACK {
        return rows;
    }
    let withheld = rows.len() - limit;
    let mut kept = rows;
    kept.truncate(limit);
    kept.push(elide_to_width(
        &format!("      … +{withheld} lines · ctrl-t"),
        width,
    ));
    kept
}

fn event_belongs_to(event: &ui::Event, id: &str) -> bool {
    if let Some(process) = &event.process {
        return process == id;
    }
    event
        .who
        .strip_prefix(id)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with(' ') || rest.starts_with('('))
}

/// One colour per process so speakers are told apart at a glance. Deliberately
/// excludes the colours that already mean something here — red is a warning,
/// dark gray is trace/summary/status furniture, cyan is the human — and skips
/// near-black and near-white indices so every entry stays legible on a dark and
/// a light terminal alike.
const PROCESS_COLORS: [Color; 14] = [
    Color::Indexed(33),  // blue
    Color::Indexed(35),  // green
    Color::Indexed(66),  // steel
    Color::Indexed(71),  // sea green
    Color::Indexed(99),  // slate blue
    Color::Indexed(105), // medium purple
    Color::Indexed(108), // sage
    Color::Indexed(129), // purple
    Color::Indexed(133), // orchid
    Color::Indexed(136), // goldenrod
    Color::Indexed(141), // lilac
    Color::Indexed(166), // dark orange
    Color::Indexed(168), // rose
    Color::Indexed(172), // amber
];

/// The selected row wears a band rather than a colour, so nothing on it has to
/// be repainted. Dark enough to sit under the palette above without competing
/// with it, light enough to be seen at a glance beside unbanded rows.
const SELECTED_BG: Color = Color::Indexed(236);

/// The tree's own rules. Structure is context, not content: the pipes were
/// drawn in the terminal's foreground colour, which made the scaffolding as
/// loud as the ids hanging off it.
const TREE_RULE: Color = Color::Indexed(240);

/// Model size reads as a temperature: cool for the cheap tier, warm for the
/// expensive one, so a rail full of large processes looks costly before a
/// single figure is read. Deliberately outside `PROCESS_COLORS` — these say
/// how big, not who.
const SIZE_COLORS: [Color; 3] = [
    Color::Indexed(107), // small — olive
    Color::Indexed(179), // medium — sand
    Color::Indexed(174), // large — clay
];

/// A process keeps its colour for the whole session and across `--resume`, so
/// the colour is derived, never drawn: an RNG would reshuffle the dashboard on
/// every restart and a per-redraw one would strobe. FNV-1a by hand rather than
/// `DefaultHasher`, which std explicitly does not promise to keep stable across
/// Rust versions — this mapping has to survive a toolchain bump.
fn process_color(id: &str) -> Color {
    let mut hash: u32 = 0x811c_9dc5;
    for byte in id.as_bytes() {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    PROCESS_COLORS[hash as usize % PROCESS_COLORS.len()]
}

/// Colour identity is the id, not the label: labels read "proc-3 worker", and a
/// process that gains or changes its name must not change colour.
fn label_id(who: &str) -> &str {
    who.split_whitespace().next().unwrap_or(who)
}

/// The colour of a speaker label, or `None` when the label does not name a
/// process — the human's own turns stay cyan and `external`/`system` rows stay
/// furniture, so colour keeps meaning "which process".
fn speaker_color(event: &ui::Event) -> Option<Color> {
    match event.kind {
        "user" | "external" | "system" => None,
        _ if event.who.is_empty() || event.who == "user" => None,
        _ => Some(process_color(label_id(&event.who))),
    }
}

/// One transcript row split into who spoke and what they said: the label wears
/// the process colour (keeping the row's own emphasis), the body keeps the
/// kind-based style that says warning, trace, or message.
fn labelled_line(
    text: String,
    label_columns: usize,
    label: Option<Color>,
    body: Style,
) -> Line<'static> {
    let Some(color) = label.filter(|_| label_columns > 0) else {
        return Line::styled(text, body);
    };
    let split = text
        .char_indices()
        .nth(label_columns)
        .map_or(text.len(), |(index, _)| index);
    let (head, tail) = text.split_at(split);
    if tail.is_empty() {
        return Line::styled(head.to_string(), body.fg(color));
    }
    Line::from(vec![
        Span::styled(head.to_string(), body.fg(color)),
        Span::styled(tail.to_string(), body),
    ])
}

fn event_style(kind: &str) -> Style {
    match kind {
        "trace" => Style::default().fg(Color::DarkGray),
        "mail" => Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD),
        "warn" => Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        "external" => Style::default().fg(Color::Yellow),
        "user" | "incoming" => Style::default().fg(Color::Cyan),
        "system" => Style::default().fg(Color::DarkGray),
        _ => Style::default().fg(Color::White),
    }
}

/// Ratatui ultimately writes cell symbols to the terminal. Never let an event
/// smuggle cursor movement, erase-screen, carriage-return, or other control
/// bytes into those symbols: one such byte invalidates the back-buffer diff.
fn terminal_safe(text: &str) -> String {
    let mut safe = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '\n' => safe.push('\n'),
            '\r' => {}
            '\t' => safe.push_str("    "),
            '\u{1b}' => safe.push('␛'),
            character if character.is_control() => safe.push('�'),
            character => safe.push(character),
        }
    }
    safe
}

fn safe_inline(text: &str) -> String {
    terminal_safe(text).replace('\n', " ")
}

/// Character wrapping keeps the newest feed content bottom-aligned without
/// relying on ratatui's unstable rendered-line-info feature.
fn wrap_chars(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    if text.is_empty() {
        return vec![String::new()];
    }
    let mut out = Vec::new();
    let mut line = String::new();
    let mut columns = 0;
    for character in text.chars() {
        if columns == width {
            out.push(std::mem::take(&mut line));
            columns = 0;
        }
        line.push(character);
        columns += 1;
    }
    if !line.is_empty() {
        out.push(line);
    }
    out
}

/// A collapsed run has to stay exactly one row, so keep the first wrapped
/// chunk and mark the elision rather than letting a long step look complete.
fn elide_to_width(text: &str, width: usize) -> String {
    let mut chunks = wrap_chars(text, width);
    if chunks.len() == 1 {
        return chunks.remove(0);
    }
    let mut line = chunks.remove(0);
    line.pop();
    line.push('…');
    line
}

#[derive(Debug, PartialEq, Eq)]
struct TreeRow {
    index: usize,
    prefix: String,
}

fn tree_rows(snapshot: &SystemSnapshot) -> Vec<TreeRow> {
    let ids: HashSet<&str> = snapshot
        .processes
        .iter()
        .map(|process| process.id.as_str())
        .collect();
    let roots: Vec<usize> = snapshot
        .processes
        .iter()
        .enumerate()
        .filter(|(_, process)| !ids.contains(process.parent.as_str()))
        .map(|(index, _)| index)
        .collect();
    let mut rows = Vec::with_capacity(snapshot.processes.len());
    let mut visited = HashSet::new();
    for (position, index) in roots.iter().copied().enumerate() {
        walk_tree(
            snapshot,
            index,
            "",
            position + 1 == roots.len(),
            true,
            &mut visited,
            &mut rows,
        );
    }
    // A malformed restored parent cycle should not make processes disappear
    // from the human's view. Append anything the rooted walk could not reach.
    for index in 0..snapshot.processes.len() {
        if !visited.contains(&index) {
            walk_tree(snapshot, index, "", true, true, &mut visited, &mut rows);
        }
    }
    rows
}

fn walk_tree(
    snapshot: &SystemSnapshot,
    index: usize,
    prefix: &str,
    last: bool,
    root: bool,
    visited: &mut HashSet<usize>,
    rows: &mut Vec<TreeRow>,
) {
    if !visited.insert(index) {
        return;
    }
    let row_prefix = if root {
        String::new()
    } else {
        format!("{prefix}{}─ ", if last { "└" } else { "├" })
    };
    rows.push(TreeRow {
        index,
        prefix: row_prefix,
    });

    let children: Vec<usize> = snapshot
        .processes
        .iter()
        .enumerate()
        .filter(|(_, process)| process.parent == snapshot.processes[index].id)
        .map(|(child, _)| child)
        .collect();
    let child_prefix = if root {
        String::new()
    } else {
        format!("{prefix}{}  ", if last { " " } else { "│" })
    };
    for (position, child) in children.iter().copied().enumerate() {
        walk_tree(
            snapshot,
            child,
            &child_prefix,
            position + 1 == children.len(),
            false,
            visited,
            rows,
        );
    }
}

fn status_glyph(status: Status) -> (&'static str, Style) {
    match status {
        Status::Running => ("●", Style::default().fg(Color::Green)),
        Status::Idle => ("○", Style::default().fg(Color::Blue)),
        Status::Stopped => ("■", Style::default().fg(Color::DarkGray)),
    }
}

/// Spinner frame for an elapsed time. Pure, and a function of wall time rather
/// than of a redraw counter: an irregular redraw then skips frames instead of
/// slowing the animation down.
fn spinner_at(elapsed: Duration) -> &'static str {
    let period = SPINNER_PERIOD.as_millis().max(1);
    SPINNER[(elapsed.as_millis() / period) as usize % SPINNER.len()]
}

/// How long a turn has been running, in as few columns as possible: seconds
/// while that is honest, then minutes and seconds, then hours and minutes.
fn format_elapsed(elapsed: Duration) -> String {
    let seconds = elapsed.as_secs();
    match seconds {
        0..=59 => format!("{seconds}s"),
        60..=3_599 => format!("{}m{:02}s", seconds / 60, seconds % 60),
        _ => format!("{}h{:02}m", seconds / 3_600, (seconds % 3_600) / 60),
    }
}

/// Money, at the precision the number deserves: a run is worth fractions of a
/// cent for a long time, and once it is worth dollars nobody reads the fourth
/// decimal.
fn format_usd(usd: f64) -> String {
    if usd >= 1.0 {
        format!("${usd:.2}")
    } else {
        format!("${usd:.4}")
    }
}

/// A cost figure that cannot lie about itself. `Estimated` means at least one
/// rate came from a baked-in or stale table, so it is marked rather than shown
/// as a fact. `Unknown` means some model had no price anywhere and its tokens
/// contributed no dollars at all — rendering that as "$0.00" would claim a run
/// was free, so the missing part is shown instead: "$?" when nothing could be
/// priced, "$0.0412+?" when part of it could.
fn format_cost(spend: Spend) -> String {
    match spend.confidence {
        Confidence::Measured => format_usd(spend.usd),
        Confidence::Estimated => format!("~{}", format_usd(spend.usd)),
        Confidence::Unknown if spend.usd == 0.0 => "$?".to_string(),
        Confidence::Unknown => format!("{}+?", format_usd(spend.usd)),
    }
}

/// Share of prompt tokens that were cheap cache hits, cumulative over the run.
/// Cache *writes* are not hits — a write costs around 1.25x a miss — so they
/// count in the denominator and not in the numerator.
///
/// `None` before anything has been read: 0% of nothing is not a fact worth
/// printing. A rate that is real but rounds to nothing reads `<1%`, since
/// integer division reporting a flat `0%` claims the cache is doing nothing
/// when it is doing a little — and the first cached turn of a long run is
/// exactly when someone is watching this number.
fn cache_share(usage: Usage) -> Option<String> {
    let prompt = usage.prompt();
    if prompt == 0 {
        return None;
    }
    let share = usage.cache_read * 100 / prompt;
    Some(if share == 0 && usage.cache_read > 0 {
        "<1%".to_string()
    } else {
        format!("{share}%")
    })
}

/// The process rail is 24-34 columns wide, so its detail row carries only the
/// two figures being watched. The model size used to compete with them for the
/// same columns and lost every time below about 34 wide; it is now a chip in
/// the column above (see `size_chip`) and costs this row nothing.
fn process_detail(tokens: u64, spend: Spend, width: usize) -> String {
    elide_to_width(
        &format!("{} ctx · {}", format_tokens(tokens), format_cost(spend)),
        width,
    )
}

/// The model size as a single column, rendered directly under the status glyph
/// on the detail row: it therefore costs the rail no width, stays legible at
/// the narrowest rail, and lines up in a column of its own however deep the
/// tree runs. A script has no model, so its column is blank rather than
/// invented; a process pinned to a named model shows that name's initial in
/// plain grey, which says "not one of the three tiers". The status line spells
/// size and effort out in full for the selected process.
fn size_chip(runs: &str) -> (String, Color) {
    match runs.split('/').next().unwrap_or_default() {
        "small" => ("S".to_string(), SIZE_COLORS[0]),
        "medium" => ("M".to_string(), SIZE_COLORS[1]),
        "large" => ("L".to_string(), SIZE_COLORS[2]),
        "script" => (" ".to_string(), Color::DarkGray),
        other => (
            other
                .chars()
                .next()
                .map(|initial| initial.to_uppercase().to_string())
                .unwrap_or_else(|| " ".to_string()),
            Color::Gray,
        ),
    }
}

fn format_tokens(tokens: u64) -> String {
    match tokens {
        0..=999 => tokens.to_string(),
        1_000..=999_999 => format!("{:.1}k", tokens as f64 / 1_000.0),
        _ => format!("{:.1}m", tokens as f64 / 1_000_000.0),
    }
}

fn process_count(count: usize) -> String {
    format!(
        "{count} {}",
        if count == 1 { "process" } else { "processes" }
    )
}

fn activity_window(
    total: usize,
    visible: usize,
    requested_scroll: usize,
) -> (std::ops::Range<usize>, usize, usize) {
    let visible = visible.max(1);
    let max_scroll = total.saturating_sub(visible);
    let scroll = requested_scroll.min(max_scroll);
    let end = total.saturating_sub(scroll);
    (end.saturating_sub(visible)..end, max_scroll, scroll)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::ProcessSnapshot;
    use ratatui::backend::TestBackend;

    fn process(id: &str, parent: &str) -> ProcessSnapshot {
        ProcessSnapshot {
            id: id.into(),
            name: None,
            parent: parent.into(),
            status: Status::Idle,
            runs: "small/low".into(),
            tokens: 0,
            spend: Default::default(),
        }
    }

    fn transcript(state: &DashboardState, width: usize) -> Vec<String> {
        activity_lines(state, width)
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect()
    }

    fn snapshot(processes: Vec<ProcessSnapshot>) -> SystemSnapshot {
        SystemSnapshot {
            processes,
            billable: 0,
            peak_context: 0,
            spend: Default::default(),
            settled: true,
        }
    }

    fn spend(usd: f64, confidence: Confidence) -> Spend {
        Spend {
            usd,
            confidence,
            usage: Usage::default(),
        }
    }

    fn status_line(state: &mut DashboardState) -> String {
        let backend = TestBackend::new(140, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| render(frame, state, "amber-otter"))
            .unwrap();
        let buffer = terminal.backend().buffer();
        let bottom = buffer.area.height - 1;
        (0..buffer.area.width)
            .map(|x| buffer[(x, bottom)].symbol())
            .collect()
    }

    #[test]
    fn tree_preserves_spawn_order_and_depth() {
        let state = snapshot(vec![
            process("proc-1", "user"),
            process("proc-2", "proc-1"),
            process("proc-3", "proc-2"),
            process("proc-4", "proc-1"),
        ]);
        let rows = tree_rows(&state);
        assert_eq!(
            rows.iter().map(|row| row.index).collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
        assert_eq!(rows[1].prefix, "├─ ");
        assert_eq!(rows[2].prefix, "│  └─ ");
        assert_eq!(rows[3].prefix, "└─ ");
    }

    #[test]
    fn process_filter_respects_id_boundaries() {
        let event = ui::Event {
            kind: "say",
            who: "proc-10 worker".into(),
            process: Some("proc-10".into()),
            text: "hello".into(),
        };
        assert!(event_belongs_to(&event, "proc-10"));
        assert!(!event_belongs_to(&event, "proc-1"));
    }

    #[test]
    fn wrapping_keeps_every_character() {
        assert_eq!(wrap_chars("abcdefgh", 3), ["abc", "def", "gh"]);
        assert_eq!(wrap_chars("", 3), [""]);
    }

    #[test]
    fn transcript_window_scrolls_back_from_the_latest_lines() {
        assert_eq!(activity_window(10, 3, 0), (7..10, 7, 0));
        assert_eq!(activity_window(10, 3, 3), (4..7, 7, 3));
        assert_eq!(activity_window(10, 3, 99), (0..3, 7, 7));
    }

    #[test]
    fn ctrl_t_toggles_traces_but_plain_t_is_just_text() {
        let mut state = DashboardState::new(snapshot(vec![process("proc-1", "user")]));
        let mut dispatch = |_: &str| false;

        assert!(!state.show_traces);
        handle_key(
            KeyEvent::new(KeyCode::Char('t'), KeyModifiers::CONTROL),
            &mut state,
            &mut dispatch,
        );
        assert!(state.show_traces);
        handle_key(
            KeyEvent::new(KeyCode::Char('t'), KeyModifiers::CONTROL),
            &mut state,
            &mut dispatch,
        );
        assert!(!state.show_traces);

        // Plain `t` is no longer a shortcut: it must land in the input like
        // any other character, even when the command line is empty.
        handle_key(
            KeyEvent::new(KeyCode::Char('t'), KeyModifiers::NONE),
            &mut state,
            &mut dispatch,
        );
        assert!(!state.show_traces);
        assert_eq!(state.input, "t");
    }

    #[test]
    fn ctrl_o_hands_the_mouse_back_to_the_terminal_and_says_so() {
        let mut state = DashboardState::new(snapshot(vec![process("proc-1", "user")]));
        let mut dispatch = |_: &str| false;

        // Capture starts on, because the wheel is worth more than selection to
        // most people most of the time.
        assert!(state.mouse_capture);
        assert!(!status_line(&mut state).contains("mouse off"));

        handle_key(
            KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL),
            &mut state,
            &mut dispatch,
        );
        assert!(!state.mouse_capture);
        let off = status_line(&mut state);
        // The state and the way out of it, in one segment.
        assert!(off.contains("mouse off"), "{off}");
        assert!(off.contains("ctrl-o"), "{off}");

        handle_key(
            KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL),
            &mut state,
            &mut dispatch,
        );
        assert!(state.mouse_capture);
        let on = status_line(&mut state);
        assert!(!on.contains("mouse off"), "{on}");
        // The binding stays discoverable even when nothing is wrong.
        assert!(on.contains("ctrl-o mouse"), "{on}");

        // A bare o types, like every other printable character.
        handle_key(
            KeyEvent::new(KeyCode::Char('o'), KeyModifiers::NONE),
            &mut state,
            &mut dispatch,
        );
        assert!(state.mouse_capture);
        assert_eq!(state.input, "o");
    }

    #[test]
    fn the_transcript_scrolls_from_the_keyboard_when_the_wheel_is_gone() {
        let mut state = DashboardState::new(snapshot(vec![process("proc-1", "user")]));
        let mut dispatch = |_: &str| false;
        state.max_scroll = 40;
        state.activity_page = 10;

        let mut press = |code: KeyCode, state: &mut DashboardState| {
            handle_key(
                KeyEvent::new(code, KeyModifiers::NONE),
                state,
                &mut dispatch,
            );
        };
        press(KeyCode::PageUp, &mut state);
        assert_eq!(state.scroll_from_bottom, 10);
        press(KeyCode::PageUp, &mut state);
        assert_eq!(state.scroll_from_bottom, 20);
        press(KeyCode::PageDown, &mut state);
        assert_eq!(state.scroll_from_bottom, 10);
        press(KeyCode::Home, &mut state);
        assert_eq!(state.scroll_from_bottom, 40);
        press(KeyCode::End, &mut state);
        assert_eq!(state.scroll_from_bottom, 0);
    }

    #[test]
    fn mouse_wheel_scrolls_transcript_without_selecting_a_process() {
        let mut state = DashboardState::new(snapshot(vec![
            process("proc-1", "user"),
            process("proc-2", "proc-1"),
        ]));
        state.selected = Some("proc-1".into());
        state.max_scroll = 20;

        handle_mouse(
            MouseEvent {
                kind: MouseEventKind::ScrollUp,
                column: 0,
                row: 0,
                modifiers: KeyModifiers::NONE,
            },
            &mut state,
        );
        assert_eq!(state.scroll_from_bottom, WHEEL_SCROLL_LINES);
        assert_eq!(state.selected.as_deref(), Some("proc-1"));

        handle_mouse(
            MouseEvent {
                kind: MouseEventKind::ScrollDown,
                column: 0,
                row: 0,
                modifiers: KeyModifiers::NONE,
            },
            &mut state,
        );
        assert_eq!(state.scroll_from_bottom, 0);
        assert_eq!(state.selected.as_deref(), Some("proc-1"));
    }

    #[test]
    fn process_count_uses_singular_and_plural_labels() {
        assert_eq!(process_count(0), "0 processes");
        assert_eq!(process_count(1), "1 process");
        assert_eq!(process_count(2), "2 processes");
    }

    #[test]
    fn terminal_controls_are_rendered_as_inert_text() {
        assert_eq!(
            terminal_safe("before\x1b[2J\r\nafter\t\x08"),
            "before␛[2J\nafter    �"
        );

        let mut state = DashboardState::new(snapshot(vec![process("proc-1", "user")]));
        state.push(ui::Event {
            kind: "external",
            who: String::new(),
            process: None,
            text: "raw\x1b[2J output".into(),
        });
        let rendered = activity_lines(&state, 80)
            .iter()
            .flat_map(|line| line.spans.iter())
            .map(|span| span.content.as_ref())
            .collect::<String>();
        assert!(rendered.contains("external output"));
        assert!(rendered.contains("raw␛[2J output"));
        assert!(!rendered.chars().any(char::is_control));
    }

    #[test]
    fn incoming_user_mail_belongs_to_its_recipient() {
        let event = ui::Event {
            kind: "incoming",
            who: "user".into(),
            process: Some("proc-10".into()),
            text: "please continue".into(),
        };
        assert!(event_belongs_to(&event, "proc-10"));
        assert!(!event_belongs_to(&event, "proc-1"));
    }

    #[test]
    fn adjacent_streamed_lines_become_one_visual_turn() {
        let mut state = DashboardState::new(snapshot(vec![process("proc-1", "user")]));
        for text in ["first", "second"] {
            state.push(ui::Event {
                kind: "say",
                who: "proc-1 root".into(),
                process: Some("proc-1".into()),
                text: text.into(),
            });
        }
        assert_eq!(state.feed.len(), 1);
        assert_eq!(state.feed[0].text, "first\nsecond");

        state.push(ui::Event {
            kind: "incoming",
            who: "user".into(),
            process: Some("proc-1".into()),
            text: "new turn".into(),
        });
        state.push(ui::Event {
            kind: "say",
            who: "proc-1 root".into(),
            process: Some("proc-1".into()),
            text: "third".into(),
        });
        assert_eq!(state.feed.len(), 3);
    }

    #[test]
    fn hidden_traces_collapse_per_speaker_into_one_summary_line() {
        let mut state = DashboardState::new(snapshot(vec![
            process("proc-1", "user"),
            process("proc-2", "proc-1"),
        ]));
        for text in [
            "→ run_script {…}",
            "  ✗ deno failed",
            "· idle, waiting for messages",
        ] {
            state.push(ui::Event {
                kind: "trace",
                who: "proc-1 root".into(),
                process: Some("proc-1".into()),
                text: text.into(),
            });
        }
        state.push(ui::Event {
            kind: "trace",
            who: "proc-2 worker".into(),
            process: Some("proc-2".into()),
            text: "⇠ mail from proc-1".into(),
        });

        assert_eq!(
            transcript(&state, 80),
            [
                "  ↳ proc-1 root  … 3 steps · ✗ deno failed",
                "  ↳ proc-2 worker  ⇠ mail from proc-1",
            ]
        );

        state.selected = Some("proc-2".into());
        assert_eq!(
            transcript(&state, 80),
            ["  ↳ proc-2 worker  ⇠ mail from proc-1"]
        );
    }

    #[test]
    fn a_block_of_collapsed_runs_is_set_off_but_not_broken_up() {
        fn trace(state: &mut DashboardState, who: &str, id: &str, text: &str) {
            state.push(ui::Event {
                kind: "trace",
                who: who.into(),
                process: Some(id.into()),
                text: text.into(),
            });
        }

        let mut state = DashboardState::new(snapshot(vec![
            process("proc-1", "user"),
            process("proc-2", "proc-1"),
        ]));
        state.push(ui::Event {
            kind: "mail",
            who: "proc-1 root".into(),
            process: Some("proc-1".into()),
            text: "starting".into(),
        });
        trace(&mut state, "proc-1 root", "proc-1", "→ run_script {…}");
        trace(&mut state, "proc-1 root", "proc-1", "· idle");
        trace(&mut state, "proc-2 worker", "proc-2", "⇠ mail from proc-1");
        state.push(ui::Event {
            kind: "say",
            who: "proc-1 root".into(),
            process: Some("proc-1".into()),
            text: "done".into(),
        });

        // Two adjacent summaries stay tight; one blank row sets the block off
        // from the message that follows it.
        assert_eq!(
            transcript(&state, 80),
            [
                "› proc-1 root → you  starting",
                "",
                "  ↳ proc-1 root  … 2 steps · · idle",
                "  ↳ proc-2 worker  ⇠ mail from proc-1",
                "",
                "• proc-1 root  done",
            ]
        );

        // Nothing follows the summary at the bottom of the feed, so nothing is
        // separated from it — no trailing blank row, same as every other kind.
        let mut ending = DashboardState::new(snapshot(vec![process("proc-1", "user")]));
        trace(&mut ending, "proc-1 root", "proc-1", "→ run_script {…}");
        assert_eq!(
            transcript(&ending, 80),
            ["  ↳ proc-1 root  → run_script {…}"]
        );

        // With traces on the expanded view must not double in height.
        ending.show_traces = true;
        trace(&mut ending, "proc-1 root", "proc-1", "· idle");
        assert_eq!(
            transcript(&ending, 80),
            [
                "  ↳ proc-1 root  → run_script {…}",
                "  ↳ proc-1 root  · idle",
            ]
        );
    }

    #[test]
    fn showing_traces_still_renders_every_step() {
        let mut state = DashboardState::new(snapshot(vec![process("proc-1", "user")]));
        state.show_traces = true;
        for text in ["first step", "second step"] {
            state.push(ui::Event {
                kind: "trace",
                who: "proc-1 root".into(),
                process: Some("proc-1".into()),
                text: text.into(),
            });
        }
        assert_eq!(
            transcript(&state, 80),
            [
                "  ↳ proc-1 root  first step",
                "  ↳ proc-1 root  second step"
            ]
        );
    }

    #[test]
    fn a_summary_prefers_a_failure_then_the_newest_step() {
        assert_eq!(collapsed_trace_text(&["  … waiting"]), "… waiting");
        assert_eq!(
            collapsed_trace_text(&["→ tool", "· idle"]),
            "… 2 steps · · idle"
        );
        assert_eq!(
            collapsed_trace_text(&["  ✗ first failure", "  ✗ later failure", "· idle"]),
            "… 3 steps · ✗ later failure"
        );
        assert_eq!(collapsed_trace_text(&["  ✗ boom\nstack trace"]), "✗ boom…");
    }

    #[test]
    fn a_summary_never_spills_onto_a_second_row() {
        assert_eq!(elide_to_width("abcdefgh", 4), "abc…");
        assert_eq!(elide_to_width("abcd", 4), "abcd");
    }

    #[test]
    fn a_long_message_is_clamped_to_a_marker_in_the_summarized_view() {
        let mut state = DashboardState::new(snapshot(vec![process("proc-1", "user")]));
        let body = (0..40)
            .map(|line| format!("line {line}"))
            .collect::<Vec<_>>()
            .join("\n");
        state.push(ui::Event {
            kind: "mail",
            who: "proc-2 worker".into(),
            process: Some("proc-2".into()),
            text: body,
        });

        let rows = transcript(&state, 80);
        // Eight body rows and the marker; the separator is the last row of the
        // feed, so it is trimmed.
        assert_eq!(rows.len(), 9);
        assert_eq!(rows[0], "› proc-2 worker → you  line 0");
        assert_eq!(rows[7], "  line 7");
        assert_eq!(rows[8], "      … +32 lines · ctrl-t");
    }

    #[test]
    fn showing_traces_renders_a_long_message_in_full() {
        let mut state = DashboardState::new(snapshot(vec![process("proc-1", "user")]));
        state.show_traces = true;
        let body = (0..40)
            .map(|line| format!("line {line}"))
            .collect::<Vec<_>>()
            .join("\n");
        state.push(ui::Event {
            kind: "mail",
            who: "proc-2 worker".into(),
            process: Some("proc-2".into()),
            text: body,
        });

        let rows = transcript(&state, 80);
        assert_eq!(rows.len(), 40);
        assert_eq!(rows[39], "  line 39");
        assert!(rows.iter().all(|row| !row.contains("ctrl-t")));
    }

    #[test]
    fn one_endless_line_clamps_by_rendered_rows_not_source_lines() {
        let mut state = DashboardState::new(snapshot(vec![process("proc-1", "user")]));
        state.push(ui::Event {
            kind: "mail",
            who: "proc-2 worker".into(),
            process: Some("proc-2".into()),
            text: "x".repeat(1200),
        });

        // 23-column prefix plus 1200 columns wraps to 31 rows at width 40.
        let rows = transcript(&state, 40);
        assert_eq!(rows.len(), 9);
        assert_eq!(rows[8], "      … +23 lines · ctrl-t");
    }

    #[test]
    fn a_body_barely_over_the_limit_keeps_all_its_rows() {
        let long = |count: usize| {
            let mut state = DashboardState::new(snapshot(vec![process("proc-1", "user")]));
            state.push(ui::Event {
                kind: "say",
                who: "proc-1 root".into(),
                process: Some("proc-1".into()),
                text: (0..count)
                    .map(|line| format!("line {line}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            });
            transcript(&state, 80)
        };
        // Ten rows would become nine; that marker does not pay for itself.
        assert_eq!(long(10).len(), 10);
        assert!(long(10).iter().all(|row| !row.contains("ctrl-t")));
        // Eleven rows become nine, so it does.
        assert_eq!(long(11).len(), 9);
        assert_eq!(long(11)[8], "      … +3 lines · ctrl-t");
    }

    #[test]
    fn warnings_and_typed_input_are_never_clamped() {
        let mut state = DashboardState::new(snapshot(vec![process("proc-1", "user")]));
        let body = (0..40)
            .map(|line| format!("line {line}"))
            .collect::<Vec<_>>()
            .join("\n");
        state.push(ui::Event {
            kind: "warn",
            who: "proc-1 root".into(),
            process: Some("proc-1".into()),
            text: body.clone(),
        });
        state.push(ui::Event {
            kind: "user",
            who: String::new(),
            process: None,
            text: body,
        });

        let rows = transcript(&state, 80);
        // Two bodies of 40 rows each, one separator between them.
        assert_eq!(rows.len(), 81);
        assert!(rows.iter().all(|row| !row.contains("ctrl-t")));
    }

    #[test]
    fn a_clamp_marker_counts_the_rows_it_withheld() {
        let rows: Vec<String> = (0..12).map(|row| row.to_string()).collect();
        let clamped = clamp_rows(rows.clone(), 8, 80);
        assert_eq!(clamped.len(), 9);
        assert_eq!(clamped[7], "7");
        assert_eq!(clamped[8], "      … +4 lines · ctrl-t");
        // Within the slack the rows are handed back untouched.
        assert_eq!(clamp_rows(rows[..10].to_vec(), 8, 80), rows[..10]);
        // A narrow pane elides the marker instead of wrapping it.
        assert_eq!(clamp_rows(rows, 8, 14)[8], "      … +4 li…");
    }

    #[test]
    fn a_process_colour_is_a_stable_hash_of_its_id() {
        // Derived, not drawn: the same id gives the same colour every call, so
        // it survives a redraw, a restart and `--resume`.
        assert_eq!(process_color("proc-7"), process_color("proc-7"));
        assert_eq!(process_color("proc-1"), Color::Indexed(166));
        assert!(PROCESS_COLORS.contains(&process_color("an-unexpectedly-long-id")));
        // A name is decoration; the id is the identity.
        assert_eq!(label_id("proc-10 worker"), "proc-10");
        assert_eq!(
            process_color(label_id("proc-10 worker")),
            process_color("proc-10")
        );
        // Two ids may share a colour — only determinism is promised.
        assert_eq!(process_color("proc-4"), process_color("proc-4"));
        assert_eq!(process_color(""), process_color(""));
    }

    #[test]
    fn colour_names_the_speaker_while_style_still_names_the_kind() {
        let mut state = DashboardState::new(snapshot(vec![process("proc-1", "user")]));
        state.push(ui::Event {
            kind: "warn",
            who: "proc-1 root".into(),
            process: Some("proc-1".into()),
            text: "boom".into(),
        });
        state.push(ui::Event {
            kind: "trace",
            who: "proc-2 worker".into(),
            process: Some("proc-2".into()),
            text: "→ run_script".into(),
        });
        state.push(ui::Event {
            kind: "user",
            who: String::new(),
            process: None,
            text: "hi".into(),
        });

        let lines = activity_lines(&state, 80);
        // A warning body stays red and bold; only its label is recoloured.
        assert_eq!(lines[0].spans[0].style.fg, Some(process_color("proc-1")));
        assert_eq!(lines[0].spans[1].style.fg, Some(Color::Red));
        assert!(
            lines[0].spans[1]
                .style
                .add_modifier
                .contains(Modifier::BOLD)
        );
        // A collapsed trace summary keeps its dark-gray body.
        assert_eq!(lines[2].spans[0].style.fg, Some(process_color("proc-2")));
        assert_eq!(lines[2].spans[1].style.fg, Some(Color::DarkGray));
        // The human is not a process: their own turn stays one unsplit cyan
        // row, styled on the line rather than per span.
        assert_eq!(lines[4].spans.len(), 1);
        assert_eq!(lines[4].style.fg, Some(Color::Cyan));
        // Splitting a row into spans must not change a single character.
        assert_eq!(
            transcript(&state, 80),
            [
                "! proc-1 root  boom",
                "",
                "  ↳ proc-2 worker  → run_script",
                "",
                "› You  hi",
            ]
        );
    }

    #[test]
    fn the_process_list_and_status_filter_use_the_same_colour_as_the_transcript() {
        let mut state = DashboardState::new(snapshot(vec![
            process("proc-1", "user"),
            process("proc-2", "proc-1"),
        ]));
        state.selected = Some("proc-1".into());
        let backend = TestBackend::new(100, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| render(frame, &mut state, "amber-otter"))
            .unwrap();

        let buffer = terminal.backend().buffer();
        let bottom = buffer.area.height - 1;
        let status: Vec<Color> = (0..buffer.area.width)
            .map(|x| buffer[(x, bottom)].fg)
            .collect();
        assert!(status.contains(&process_color("proc-1")));
        // proc-2 is not selected, so its list entry shows its own colour.
        let listed = (0..bottom)
            .any(|y| (0..buffer.area.width).any(|x| buffer[(x, y)].fg == process_color("proc-2")));
        assert!(listed);
    }

    #[test]
    fn cost_and_cache_figures_are_honest_about_what_they_know() {
        // A process that has made no requests really has spent nothing.
        assert_eq!(format_cost(spend(0.0, Confidence::Measured)), "$0.0000");
        assert_eq!(format_cost(spend(0.0412, Confidence::Measured)), "$0.0412");
        // Dollars do not need four decimals.
        assert_eq!(format_cost(spend(1.5, Confidence::Measured)), "$1.50");
        // A guess is always marked as one.
        assert_eq!(
            format_cost(spend(0.0412, Confidence::Estimated)),
            "~$0.0412"
        );
        // An unpriced model contributed no dollars, so a zero would be a lie.
        assert_eq!(format_cost(spend(0.0, Confidence::Unknown)), "$?");
        assert_eq!(format_cost(spend(0.0412, Confidence::Unknown)), "$0.0412+?");
        assert!(!format_cost(spend(0.0, Confidence::Unknown)).contains("0.00"));
        // Every case reads differently from every other.
        let rendered = [
            format_cost(spend(0.0412, Confidence::Measured)),
            format_cost(spend(0.0412, Confidence::Estimated)),
            format_cost(spend(0.0412, Confidence::Unknown)),
            format_cost(spend(0.0, Confidence::Measured)),
            format_cost(spend(0.0, Confidence::Unknown)),
        ];
        assert_eq!(
            rendered.iter().collect::<HashSet<_>>().len(),
            rendered.len()
        );

        assert_eq!(cache_share(Usage::default()), None);
        assert_eq!(
            cache_share(Usage {
                uncached_input: 100,
                cache_write: 0,
                cache_read: 300,
                output: 50,
            })
            .as_deref(),
            Some("75%")
        );
    }

    #[test]
    fn a_real_cache_hit_rate_never_rounds_down_to_nothing() {
        let prompt = |uncached_input, cache_write, cache_read| {
            cache_share(Usage {
                uncached_input,
                cache_write,
                cache_read,
                output: 0,
            })
        };
        // Nothing read yet: the segment is omitted rather than shown as zero.
        assert_eq!(prompt(0, 0, 0), None);
        // Read, but nothing hit: a true zero.
        assert_eq!(prompt(1_000, 0, 0).as_deref(), Some("0%"));
        // One hit in ten thousand is not nothing, and integer division said it
        // was.
        assert_eq!(prompt(9_999, 0, 1).as_deref(), Some("<1%"));
        assert_eq!(prompt(50, 0, 50).as_deref(), Some("50%"));
        // A write is not a hit: it costs more than a miss, so it stays in the
        // denominator and out of the numerator.
        assert_eq!(prompt(0, 100, 0).as_deref(), Some("0%"));
        assert_eq!(prompt(0, 100, 100).as_deref(), Some("50%"));
    }

    #[test]
    fn the_rail_keeps_context_and_cost_and_sizes_them_in_a_column() {
        assert_eq!(process_detail(0, Spend::default(), 31), "0 ctx · $0.0000");
        assert_eq!(
            process_detail(12_400, spend(0.0412, Confidence::Estimated), 24),
            "12.4k ctx · ~$0.0412"
        );
        // Narrow: elide rather than spill onto a second row.
        assert_eq!(
            process_detail(12_400, spend(0.0412, Confidence::Estimated), 12),
            "12.4k ctx ·…"
        );

        // The size is a column, not a field, so it survives every width. Each
        // tier reads differently in both glyph and colour.
        let tiers: Vec<(String, Color)> = ["small/low", "medium/high", "large"]
            .into_iter()
            .map(size_chip)
            .collect();
        assert_eq!(
            tiers
                .iter()
                .map(|(chip, _)| chip.as_str())
                .collect::<Vec<_>>(),
            ["S", "M", "L"]
        );
        assert_eq!(
            tiers
                .iter()
                .map(|(_, color)| color)
                .collect::<HashSet<_>>()
                .len(),
            3
        );
        // A script has no model size; inventing one would be a lie.
        assert_eq!(size_chip("script").0, " ");
        // A pinned model is not a tier, and says so in grey.
        assert_eq!(size_chip("opus-4/max"), ("O".to_string(), Color::Gray));
    }

    #[test]
    fn the_status_line_shows_the_selected_process_context_not_the_system_peak() {
        let mut processes = vec![process("proc-1", "user"), process("proc-2", "proc-1")];
        processes[0].tokens = 12_400;
        processes[1].tokens = 2_000;
        let mut system = snapshot(processes);
        system.peak_context = 999_000;
        let mut state = DashboardState::new(system);

        let first = status_line(&mut state);
        assert!(first.contains("12.4k ctx"), "{first}");
        assert!(!first.contains("999.0k"), "{first}");

        state.selected = Some("proc-2".into());
        let second = status_line(&mut state);
        assert!(second.contains("2.0k ctx"), "{second}");
        assert!(!second.contains("12.4k"), "{second}");
    }

    #[test]
    fn the_header_carries_the_run_totals_the_status_line_no_longer_shows() {
        let mut system = snapshot(vec![process("proc-1", "user")]);
        // Nothing has run yet: no zeroes to read past.
        assert_eq!(system_summary(&system), "  ·  1 process");

        system.billable = 1_200_000;
        system.peak_context = 48_100;
        system.spend = Spend {
            usd: 0.62,
            confidence: Confidence::Measured,
            usage: Usage {
                uncached_input: 100,
                cache_write: 0,
                cache_read: 300,
                output: 50,
            },
        };
        assert_eq!(
            system_summary(&system),
            "  ·  1 process  ·  run 75% cache hits  ·  1.2m billable  ·  peak 48.1k ctx"
        );
    }

    #[test]
    fn an_unpriced_run_total_never_reads_as_zero_dollars() {
        let mut processes = vec![process("proc-1", "user")];
        processes[0].spend = spend(0.0412, Confidence::Estimated);
        let mut system = snapshot(processes);
        system.spend = spend(0.0, Confidence::Unknown);
        let mut state = DashboardState::new(system);

        let line = status_line(&mut state);
        assert!(line.contains("cost ~$0.0412"), "{line}");
        assert!(line.contains("run $?"), "{line}");
        assert!(!line.contains("$0.00"), "{line}");
    }

    #[test]
    fn the_spinner_advances_with_wall_time_and_wraps() {
        assert_eq!(spinner_at(Duration::ZERO), "⠋");
        assert_eq!(spinner_at(Duration::from_millis(119)), "⠋");
        assert_eq!(spinner_at(Duration::from_millis(120)), "⠙");
        assert_eq!(spinner_at(Duration::from_millis(240)), "⠹");
        // Ten frames on, it is back where it started.
        assert_eq!(spinner_at(Duration::from_millis(1_200)), "⠋");
        assert_eq!(spinner_at(Duration::from_secs(3_600)), "⠋");
    }

    #[test]
    fn elapsed_time_stays_compact_as_a_turn_drags_on() {
        assert_eq!(format_elapsed(Duration::ZERO), "0s");
        assert_eq!(format_elapsed(Duration::from_millis(9_400)), "9s");
        assert_eq!(format_elapsed(Duration::from_secs(59)), "59s");
        assert_eq!(format_elapsed(Duration::from_secs(60)), "1m00s");
        assert_eq!(format_elapsed(Duration::from_secs(61)), "1m01s");
        assert_eq!(format_elapsed(Duration::from_secs(3_599)), "59m59s");
        assert_eq!(format_elapsed(Duration::from_secs(3_600)), "1h00m");
        assert_eq!(format_elapsed(Duration::from_secs(7_325)), "2h02m");
    }

    #[test]
    fn a_turn_is_timed_from_when_it_started_and_forgotten_when_it_ends() {
        let mut working = vec![process("proc-1", "user"), process("proc-2", "proc-1")];
        working[0].status = Status::Running;
        let mut state = DashboardState::new(snapshot(working.clone()));
        state.refresh(snapshot(working.clone()));
        let first = state.working_since["proc-1"];
        assert!(state.working_for("proc-1").is_some());
        // An idle process is never timed.
        assert!(state.working_for("proc-2").is_none());

        // Still the same turn: the clock is not restarted by a redraw.
        state.refresh(snapshot(working.clone()));
        assert_eq!(state.working_since["proc-1"], first);

        // The turn ends, and with it the timer.
        let mut idle = working.clone();
        idle[0].status = Status::Idle;
        state.refresh(snapshot(idle));
        assert!(state.working_for("proc-1").is_none());

        // A new turn starts from zero rather than from the first one.
        state.refresh(snapshot(working));
        assert!(state.working_since["proc-1"] > first);
    }

    #[test]
    fn a_working_process_spins_in_the_rail_and_the_status_line() {
        let mut processes = vec![process("proc-1", "user"), process("proc-2", "proc-1")];
        processes[0].status = Status::Running;
        let mut state = DashboardState::new(snapshot(processes));
        state.working_since.insert(
            "proc-1".into(),
            Instant::now() - Duration::from_millis(12_500),
        );

        let backend = TestBackend::new(120, 16);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| render(frame, &mut state, "amber-otter"))
            .unwrap();
        let buffer = terminal.backend().buffer();
        let rail: Vec<String> = (0..buffer.area.height)
            .map(|y| (0..30).map(|x| buffer[(x, y)].symbol()).collect())
            .collect();
        let rail = rail.join("\n");

        assert!(
            SPINNER
                .iter()
                .any(|frame| rail.contains(&format!("{frame} proc-1"))),
            "{rail}"
        );
        assert!(rail.contains("12s"), "{rail}");
        // An idle process keeps its static glyph and gets no stopwatch.
        assert!(rail.contains("○ proc-2"), "{rail}");

        let line = status_line(&mut state);
        assert!(SPINNER.iter().any(|frame| line.contains(frame)), "{line}");
        assert!(line.contains("12s"), "{line}");
    }

    #[test]
    fn the_selected_row_keeps_its_own_colours_under_the_highlight() {
        let mut processes = vec![process("proc-1", "user"), process("proc-2", "proc-1")];
        processes[1].name = Some("msgclamp".into());
        processes[1].runs = "large/high".into();
        let mut state = DashboardState::new(snapshot(processes));
        state.selected = Some("proc-2".into());

        let backend = TestBackend::new(100, 16);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| render(frame, &mut state, "amber-otter"))
            .unwrap();
        let buffer = terminal.backend().buffer();
        // The rail is a quarter of the frame, clamped to 24-34 columns.
        let rail = 24u16;
        let cells: Vec<Vec<String>> = (0..buffer.area.height)
            .map(|y| {
                (0..rail)
                    .map(|x| buffer[(x, y)].symbol().to_string())
                    .collect()
            })
            .collect();
        let at = |row: &Vec<String>, needle: &str| -> Option<usize> {
            let width = needle.chars().count();
            (0..row.len()).find(|start| {
                row[*start..]
                    .iter()
                    .take(width)
                    .cloned()
                    .collect::<String>()
                    == needle
            })
        };
        let (y, x) = cells
            .iter()
            .enumerate()
            .find_map(|(y, row)| at(row, "proc-2").map(|x| (y, x)))
            .expect("the selected process is in the rail");
        let render = cells
            .iter()
            .map(|row| row.concat())
            .collect::<Vec<_>>()
            .join("\n");

        // Identity survives selection: the id is still the process's colour,
        // and the band — not a repaint — is what marks the row.
        let id = &buffer[(x as u16, y as u16)];
        assert_eq!(id.fg, process_color("proc-2"), "{render}");
        assert_eq!(id.bg, SELECTED_BG, "{render}");
        assert_ne!(id.fg, Color::Cyan, "{render}");
        // The band runs the width of the list, both rows of the item.
        assert_eq!(buffer[(1, y as u16)].bg, SELECTED_BG, "{render}");
        assert_eq!(buffer[(1, y as u16 + 1)].bg, SELECTED_BG, "{render}");
        // ... and stops there.
        assert_ne!(buffer[(1, y as u16 - 1)].bg, SELECTED_BG, "{render}");

        // One marker, on the row the human is pointing at — the detail row
        // below it used to carry a second copy of the same arrow.
        assert_eq!(cells[y][1], "›", "{render}");
        assert_eq!(cells[y + 1][1], " ", "{render}");
        let arrows = cells[3..y + 4]
            .iter()
            .flatten()
            .filter(|cell| cell.as_str() == "›")
            .count();
        assert_eq!(arrows, 1, "{render}");

        // The model size sits under the status glyph, in its tier's colour, on
        // the selected row and the unselected one alike.
        let glyph = x - 2;
        assert_eq!(cells[y + 1][glyph], "L", "{render}");
        assert_eq!(buffer[(glyph as u16, y as u16 + 1)].fg, SIZE_COLORS[2]);
        let (root, root_x) = cells
            .iter()
            .enumerate()
            .find_map(|(y, row)| at(row, "proc-1").map(|x| (y, x)))
            .expect("the root process is in the rail");
        assert_eq!(cells[root + 1][root_x - 2], "S", "{render}");
        assert_eq!(
            buffer[(root_x as u16 - 2, root as u16 + 1)].fg,
            SIZE_COLORS[0]
        );

        // Divider hierarchy: the rail's edge is heavy, the tree's rules are
        // thin and dim, the fields on a detail row are parted by a dot.
        assert_eq!(buffer[(rail, y as u16)].symbol(), "┃", "{render}");
        let branch = cells[y][2..x]
            .iter()
            .position(|cell| cell == "├" || cell == "└")
            .map(|offset| offset + 2)
            .expect("a child hangs off a branch");
        assert!(
            matches!(
                buffer[(branch as u16, y as u16)].fg,
                Color::DarkGray | TREE_RULE
            ),
            "{render}"
        );
        assert!(render.contains("ctx · "), "{render}");
    }

    /// The normal test harness is not a terminal, so this becomes active in
    /// the pseudo-terminal smoke run. Keeping it here makes the fd-level
    /// quarantine independently reproducible without a model/API fixture.
    #[cfg(unix)]
    #[test]
    fn direct_process_output_is_quarantined_when_attached_to_a_terminal() {
        use std::io::IsTerminal;

        if !std::io::stdout().is_terminal() {
            return;
        }

        let mut events = ui::tap().subscribe();
        ui::set_dashboard_active(true);
        let (guard, renderer) = TerminalGuard::enter().unwrap();
        let stdout = b"ROGUE_STDOUT\x1b[2J\n";
        let stderr = b"ROGUE_STDERR\n";
        // SAFETY: the guard has installed valid writable descriptors 1 and 2;
        // both byte slices remain alive for the duration of their calls.
        unsafe {
            libc::write(libc::STDOUT_FILENO, stdout.as_ptr().cast(), stdout.len());
            libc::write(libc::STDERR_FILENO, stderr.as_ptr().cast(), stderr.len());
        }

        let deadline = Instant::now() + Duration::from_secs(1);
        let mut captured = Vec::new();
        while Instant::now() < deadline && captured.len() < 2 {
            while let Ok(event) = events.try_recv() {
                if event.kind == "external" {
                    captured.push(event.text);
                }
            }
            std::thread::sleep(Duration::from_millis(5));
        }

        drop(renderer);
        drop(guard);
        ui::set_dashboard_active(false);
        assert!(captured.iter().any(|line| line.contains("ROGUE_STDOUT")));
        assert!(captured.iter().any(|line| line.contains("ROGUE_STDERR")));
    }

    #[test]
    fn frame_is_transcript_first_with_a_codex_style_composer() {
        let snapshot = snapshot(vec![process("proc-1", "user")]);
        let mut state = DashboardState::new(snapshot);
        state.push(ui::Event {
            kind: "incoming",
            who: "user".into(),
            process: Some("proc-1".into()),
            text: "ship it".into(),
        });
        let backend = TestBackend::new(100, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| render(frame, &mut state, "amber-otter"))
            .unwrap();

        let buffer = terminal.backend().buffer();
        let mut screen = String::new();
        for y in 0..buffer.area.height {
            for x in 0..buffer.area.width {
                screen.push_str(buffer[(x, y)].symbol());
            }
            screen.push('\n');
        }
        assert!(screen.contains("bitty"));
        assert!(screen.contains("1 process"));
        assert!(!screen.contains("1 processes"));
        assert!(screen.contains("› You → proc-1  ship it"));
        assert!(screen.contains("Send a message or /command"));
        assert!(screen.contains("small/low"));
        assert!(
            !screen.contains("Activity"),
            "transcript should not look like a panel"
        );
    }
}
