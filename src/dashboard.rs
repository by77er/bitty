//! Alternate-screen dashboard for `--tui`.
//!
//! Rendering is intentionally downstream of `ui`'s structured line tap and
//! `System::snapshot()`: the dashboard observes the harness, but it is not a
//! second logging or command implementation.

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
use std::collections::{HashSet, VecDeque};
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
        self.snapshot = snapshot;
        self.animation_frame = !self.animation_frame;
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
    let (_guard, output) = TerminalGuard::enter()?;
    let backend = CrosstermBackend::new(output);
    let mut terminal = Terminal::new(backend).context("could not initialize the dashboard")?;

    let mut state = DashboardState::new(system.snapshot());
    let mut input_events = EventStream::new();
    let mut snapshot_tick = tokio::time::interval(SNAPSHOT_TICK);
    let mut redraw_tick = tokio::time::interval(REDRAW_DEBOUNCE);
    snapshot_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    redraw_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut tap_open = true;
    let mut dirty = true;

    loop {
        tokio::select! {
            terminal_event = input_events.next() => {
                match terminal_event {
                    Some(Ok(TerminalEvent::Key(key))) => {
                        if handle_key(key, &mut state, &mut dispatch) {
                            break;
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
        // `t` is the promised one-key trace toggle when the command line is
        // empty. Once somebody is composing a command it is ordinary text;
        // Ctrl-T remains an unconditional toggle.
        KeyCode::Char('t') if state.input.is_empty() => {
            state.show_traces = !state.show_traces;
            state.latest();
        }
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
                format!("  ·  {}", process_count(state.snapshot.processes.len())),
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

fn render_processes(frame: &mut Frame<'_>, area: Rect, state: &DashboardState) {
    let rows = tree_rows(&state.snapshot);
    let items: Vec<ListItem<'_>> = rows
        .iter()
        .map(|row| {
            let process = &state.snapshot.processes[row.index];
            let (glyph, glyph_style) = status_glyph(process.status);
            let name = process
                .name
                .as_ref()
                .map(|name| format!(" {}", safe_inline(name)))
                .unwrap_or_default();
            let detail_indent = " ".repeat(row.prefix.chars().count() + 2);
            ListItem::new(Text::from(vec![
                Line::from(vec![
                    Span::raw(row.prefix.clone()),
                    Span::styled(format!("{glyph} "), glyph_style),
                    Span::styled(
                        process.id.clone(),
                        Style::default().add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(name),
                ]),
                Line::styled(
                    format!(
                        "{detail_indent}{} · {} ctx",
                        safe_inline(&process.runs),
                        format_tokens(process.tokens)
                    ),
                    Style::default().fg(Color::DarkGray),
                ),
            ]))
        })
        .collect();

    let selected = state.selected.as_ref().and_then(|id| {
        rows.iter()
            .position(|row| state.snapshot.processes[row.index].id == *id)
    });
    let mut list_state = ListState::default();
    list_state.select(selected);
    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::RIGHT)
                .border_style(Style::default().fg(Color::DarkGray))
                .padding(Padding::new(1, 1, 1, 0))
                .title(Span::styled(
                    " processes ",
                    Style::default().fg(Color::DarkGray),
                )),
        )
        .highlight_symbol("› ")
        .highlight_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .repeat_highlight_symbol(true);
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
    let filter = state
        .selected
        .as_ref()
        .map(|id| format!(" · filter {id}"))
        .unwrap_or_default();
    let traces = if state.show_traces {
        "traces on"
    } else {
        "traces hidden"
    };
    let scroll = if state.scroll_from_bottom == 0 {
        String::new()
    } else {
        format!(" · {} lines back · End latest", state.scroll_from_bottom)
    };
    frame.render_widget(
        Paragraph::new(Line::styled(
            format!(
                " {runs} · {} context · {} billable · {}{filter}{scroll} · {traces} · ↑↓ agents · esc clear · t toggle",
                format_tokens(state.snapshot.peak_context),
                format_tokens(state.snapshot.billable),
                safe_inline(session),
            ),
            Style::default().fg(Color::DarkGray),
        )),
        area,
    );
}

fn activity_lines(state: &DashboardState, width: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for event in state.feed.iter().filter(|event| {
        (state.show_traces || event.kind != "trace")
            && state
                .selected
                .as_deref()
                .is_none_or(|id| event_belongs_to(event, id))
    }) {
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
            "trace" if !event.who.is_empty() => format!("  ↳ {who}  "),
            "mail" if !event.who.is_empty() => format!("› {who} → you  "),
            "warn" if !event.who.is_empty() => format!("! {who}  "),
            "external" => "  ↳ external output  ".to_string(),
            "system" => "  ".to_string(),
            _ if !event.who.is_empty() => format!("• {who}  "),
            _ => "  ".to_string(),
        };
        let style = event_style(event.kind);
        for (index, source_line) in event.text.split('\n').enumerate() {
            let lead = if index == 0 {
                prefix.as_str()
            } else if prefix.is_empty() {
                ""
            } else {
                "  "
            };
            for wrapped in wrap_chars(&format!("{lead}{source_line}"), width) {
                lines.push(Line::styled(wrapped, style));
            }
        }
        if matches!(event.kind, "user" | "incoming" | "say" | "mail" | "warn") {
            lines.push(Line::default());
        }
    }
    lines
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
        }
    }

    fn snapshot(processes: Vec<ProcessSnapshot>) -> SystemSnapshot {
        SystemSnapshot {
            processes,
            billable: 0,
            peak_context: 0,
            settled: true,
        }
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
