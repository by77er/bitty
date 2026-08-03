//! Script processes: a TypeScript actor running on an embedded Deno runtime.
//!
//! Same actor contract as an agent process — an id, a mailbox, links, grants,
//! a namespace — but the behavior is deterministic code instead of a model, so
//! a script process costs no API tokens at all. That makes them the right node
//! for the mechanical parts of a topology: routing, aggregation, validation,
//! rate limiting, format conversion.
//!
//! V8 isolates are not `Send`, so each script owns a dedicated OS thread with
//! its own current-thread tokio runtime. Anything the script asks the system to
//! do is dispatched through `actions`, so an agent and a script are policed by
//! exactly the same capability and visibility code.

use crate::actions;
use crate::system::{Control, Mail, Meta, Priority, Status, System};
use crate::ui;
use deno_core::{Extension, JsRuntime, OpDecl, OpState, PollEventLoopOptions, RuntimeOptions, op2};
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedReceiver;

/// What the ops need to act on the system on the script's behalf.
struct Host {
    sys: Arc<System>,
    me: Meta,
}

fn host_call(
    state: &mut OpState,
    f: impl FnOnce(&Arc<System>, &Meta) -> (String, bool),
) -> Result<String, deno_error::JsErrorBox> {
    let host = state.borrow::<Host>();
    let (detail, is_error) = f(&host.sys, &host.me);
    if is_error {
        Err(deno_error::JsErrorBox::type_error(detail))
    } else {
        Ok(detail)
    }
}

#[op2]
#[string]
fn op_bitty_send(
    state: &mut OpState,
    #[serde] to: Vec<String>,
    #[string] message: String,
    #[string] priority: String,
) -> Result<String, deno_error::JsErrorBox> {
    let priority = if priority == "low" {
        Priority::Low
    } else {
        Priority::High
    };
    host_call(state, |sys, me| {
        actions::send(sys, me, to, &message, priority)
    })
}

#[op2]
#[string]
fn op_bitty_stop(
    state: &mut OpState,
    #[serde] targets: Vec<String>,
    cascade: bool,
) -> Result<String, deno_error::JsErrorBox> {
    host_call(state, |sys, me| actions::stop(sys, me, targets, cascade))
}

#[op2]
#[string]
fn op_bitty_list(state: &mut OpState) -> Result<String, deno_error::JsErrorBox> {
    host_call(state, |sys, me| (actions::list(sys, me), false))
}

fn fs_error(e: String) -> deno_error::JsErrorBox {
    deno_error::JsErrorBox::type_error(e)
}

#[op2]
#[string]
fn op_bitty_fs_read(
    state: &mut OpState,
    #[string] path: String,
) -> Result<String, deno_error::JsErrorBox> {
    let host = state.borrow::<Host>();
    let resolved = host.me.grants.read.resolve(&path, false).map_err(fs_error)?;
    std::fs::read_to_string(&resolved).map_err(|e| fs_error(format!("{path}: {e}")))
}

#[op2]
#[string]
fn op_bitty_fs_write(
    state: &mut OpState,
    #[string] path: String,
    #[string] contents: String,
) -> Result<String, deno_error::JsErrorBox> {
    let host = state.borrow::<Host>();
    let resolved = host.me.grants.write.resolve(&path, true).map_err(fs_error)?;
    std::fs::write(&resolved, contents).map_err(|e| fs_error(format!("{path}: {e}")))?;
    Ok(resolved.display().to_string())
}

/// Directory entries as JSON, so a script can build glob or grep on top
/// without needing an op per traversal primitive.
#[op2]
#[string]
fn op_bitty_fs_list(
    state: &mut OpState,
    #[string] path: String,
) -> Result<String, deno_error::JsErrorBox> {
    let host = state.borrow::<Host>();
    let resolved = host.me.grants.read.resolve(&path, false).map_err(fs_error)?;
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(&resolved).map_err(|e| fs_error(format!("{path}: {e}")))? {
        let entry = entry.map_err(|e| fs_error(e.to_string()))?;
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        entries.push(json!({
            "name": entry.file_name().to_string_lossy(),
            "path": entry.path().display().to_string(),
            "dir": is_dir,
        }));
    }
    Ok(Value::Array(entries).to_string())
}

#[op2]
#[string]
fn op_bitty_fs_mkdir(
    state: &mut OpState,
    #[string] path: String,
) -> Result<String, deno_error::JsErrorBox> {
    let host = state.borrow::<Host>();
    let resolved = host.me.grants.write.resolve(&path, true).map_err(fs_error)?;
    std::fs::create_dir_all(&resolved).map_err(|e| fs_error(format!("{path}: {e}")))?;
    Ok(resolved.display().to_string())
}

#[op2]
#[string]
fn op_bitty_fs_remove(
    state: &mut OpState,
    #[string] path: String,
) -> Result<String, deno_error::JsErrorBox> {
    let host = state.borrow::<Host>();
    let resolved = host.me.grants.write.resolve(&path, false).map_err(fs_error)?;
    let outcome = if resolved.is_dir() {
        std::fs::remove_dir_all(&resolved)
    } else {
        std::fs::remove_file(&resolved)
    };
    outcome.map_err(|e| fs_error(format!("{path}: {e}")))?;
    Ok(resolved.display().to_string())
}

/// Run a program. Never through a shell: the program is executed directly and
/// arguments are passed as a vector, so there is no string for a shell to
/// re-parse and no metacharacter to smuggle. The allowlist is on the program
/// name, which is a genuinely weaker boundary than a path prefix — whatever
/// runs is bounded by the OS, not by this harness.
#[op2]
#[string]
fn op_bitty_exec(
    state: &mut OpState,
    #[string] program: String,
    #[serde] args: Vec<String>,
    #[string] cwd: String,
) -> Result<String, deno_error::JsErrorBox> {
    let host = state.borrow::<Host>();
    if !host.me.may(crate::grants::Capability::Run, &program) {
        return Err(fs_error(format!(
            "not permitted to run '{program}'; you may run {}",
            host.me.permitted(crate::grants::Capability::Run)
        )));
    }
    // The working directory has to be somewhere this process can already read,
    // so running a command cannot become a way to reach outside its roots.
    let dir = host.me.grants.read.resolve(&cwd, false).map_err(fs_error)?;
    let output = std::process::Command::new(&program)
        .args(&args)
        .current_dir(&dir)
        .output()
        .map_err(|e| fs_error(format!("{program}: {e}")))?;
    Ok(json!({
        "code": output.status.code(),
        "stdout": String::from_utf8_lossy(&output.stdout),
        "stderr": String::from_utf8_lossy(&output.stderr),
    })
    .to_string())
}

/// Fetch over HTTP. The host is checked against the grant, and the request is
/// run on the *main* runtime rather than this script's, so blocking here
/// stalls only this isolate — which is correct anyway, since an actor handles
/// one message at a time.
#[op2]
#[string]
fn op_bitty_fetch(
    state: &mut OpState,
    #[string] url: String,
    #[string] method: String,
    #[string] body: String,
) -> Result<String, deno_error::JsErrorBox> {
    let host = state.borrow::<Host>();
    let parsed = reqwest::Url::parse(&url).map_err(|e| fs_error(format!("{url}: {e}")))?;
    let Some(hostname) = parsed.host_str().map(String::from) else {
        return Err(fs_error(format!("{url} names no host")));
    };
    // A grant may name the bare host or host:port; either admits the request.
    let with_port = parsed
        .port()
        .map(|p| format!("{hostname}:{p}"))
        .unwrap_or_else(|| hostname.clone());
    if !host.me.may(crate::grants::Capability::Net, &hostname)
        && !host.me.may(crate::grants::Capability::Net, &with_port)
    {
        return Err(fs_error(format!(
            "not permitted to reach '{hostname}'; you may reach {}",
            host.me.permitted(crate::grants::Capability::Net)
        )));
    }

    let client = reqwest::Client::new();
    let (tx, rx) = std::sync::mpsc::channel();
    host.sys.rt().spawn(async move {
        let request = match method.to_uppercase().as_str() {
            "POST" => client.post(parsed).body(body),
            "PUT" => client.put(parsed).body(body),
            "DELETE" => client.delete(parsed),
            _ => client.get(parsed),
        };
        let outcome = match request.send().await {
            Ok(response) => {
                let status = response.status().as_u16();
                match response.text().await {
                    Ok(text) => Ok((status, text)),
                    Err(e) => Err(e.to_string()),
                }
            }
            Err(e) => Err(e.to_string()),
        };
        let _ = tx.send(outcome);
    });
    match rx.recv() {
        Ok(Ok((status, text))) => Ok(json!({"status": status, "body": text}).to_string()),
        Ok(Err(e)) => Err(fs_error(e)),
        Err(_) => Err(fs_error("the request was dropped".into())),
    }
}

#[op2]
#[string]
fn op_bitty_env(
    state: &mut OpState,
    #[string] name: String,
) -> Result<String, deno_error::JsErrorBox> {
    let host = state.borrow::<Host>();
    if !host.me.may(crate::grants::Capability::Env, &name) {
        return Err(fs_error(format!(
            "not permitted to read '{name}'; you may read {}",
            host.me.permitted(crate::grants::Capability::Env)
        )));
    }
    Ok(std::env::var(&name).unwrap_or_default())
}

#[op2]
#[string]
fn op_bitty_sys(
    state: &mut OpState,
    #[string] key: String,
) -> Result<String, deno_error::JsErrorBox> {
    let host = state.borrow::<Host>();
    if !host.me.may(crate::grants::Capability::Sys, &key) {
        return Err(fs_error(format!(
            "not permitted to query '{key}'; you may query {}",
            host.me.permitted(crate::grants::Capability::Sys)
        )));
    }
    let value = match key.as_str() {
        "hostname" => std::env::var("HOSTNAME").unwrap_or_default(),
        "osRelease" => std::env::consts::OS.to_string(),
        "arch" => std::env::consts::ARCH.to_string(),
        "cwd" => std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_default(),
        other => return Err(fs_error(format!("unknown system key '{other}'"))),
    };
    Ok(value)
}

/// Fail a blocked caller. Distinct from `op_bitty_reply` so a thrown script
/// surfaces as an error rather than as a successful result whose text happens
/// to begin with "error".
#[op2(fast)]
fn op_bitty_reply_error(state: &mut OpState, #[string] id: String, #[string] message: String) {
    let host = state.borrow::<Host>();
    host.sys.resolve_call(&id, Err(message));
}

/// Hand a value back to a caller blocked inside `call_process`.
#[op2(fast)]
fn op_bitty_reply(state: &mut OpState, #[string] id: String, #[string] value: String) {
    let host = state.borrow::<Host>();
    host.sys.resolve_call(&id, Ok(value));
}

#[op2(fast)]
fn op_bitty_log(state: &mut OpState, #[string] text: String) {
    let host = state.borrow::<Host>();
    ui::say(&host.me.tag, &text);
}

/// The API surface the script sees. Deliberately small and mirrors the agent's
/// tools one-for-one, so the two process types are described the same way.
const PRELUDE: &str = r#"
globalThis.bitty = (() => {
  const ops = Deno.core.ops;
  let handler = null;
  const api = {
    onMail(fn) { handler = fn; },
    send(to, message, priority = "high") {
      return ops.op_bitty_send(Array.isArray(to) ? to : [to], String(message), priority);
    },
    stop(targets, cascade = false) {
      return ops.op_bitty_stop(Array.isArray(targets) ? targets : [targets], !!cascade);
    },
    list() { return ops.op_bitty_list(); },
    // Filesystem access is whatever this process was granted; every call is
    // canonicalized and checked against those roots, and throws otherwise.
    fs: {
      read(path) { return ops.op_bitty_fs_read(String(path)); },
      write(path, contents) { return ops.op_bitty_fs_write(String(path), String(contents)); },
      list(path) { return JSON.parse(ops.op_bitty_fs_list(String(path))); },
      mkdir(path) { return ops.op_bitty_fs_mkdir(String(path)); },
      remove(path) { return ops.op_bitty_fs_remove(String(path)); },
    },
    // Run a program directly — no shell, so arguments are a list, not a
    // command line. Returns {code, stdout, stderr}.
    // HTTP against hosts this process was granted. Returns {status, body}.
    fetch(url, opts = {}) {
      return JSON.parse(ops.op_bitty_fetch(
        String(url), String(opts.method ?? "GET"), String(opts.body ?? "")));
    },
    env(name) { return ops.op_bitty_env(String(name)); },
    sys(key) { return ops.op_bitty_sys(String(key)); },
    exec(program, args = [], cwd = ".") {
      return JSON.parse(ops.op_bitty_exec(String(program), args.map(String), String(cwd)));
    },
    log(text) { ops.op_bitty_log(String(text)); },
  };
  // deno_core supplies Deno.core (the ops bridge) but not the Deno namespace,
  // which lives in deno_runtime. Rather than expecting a model to unlearn the
  // API it knows, map the common surface onto our ops — every call still goes
  // through the same capability checks, and a denial still throws.
  const D = globalThis.Deno;
  D.readTextFile  = (p) => Promise.resolve(api.fs.read(p));
  D.readTextFileSync = (p) => api.fs.read(p);
  D.writeTextFile = (p, c) => Promise.resolve(api.fs.write(p, c)).then(() => undefined);
  D.writeTextFileSync = (p, c) => { api.fs.write(p, c); };
  D.mkdir     = (p) => Promise.resolve(api.fs.mkdir(p)).then(() => undefined);
  D.mkdirSync = (p) => { api.fs.mkdir(p); };
  D.remove     = (p) => Promise.resolve(api.fs.remove(p)).then(() => undefined);
  D.removeSync = (p) => { api.fs.remove(p); };
  // Deno.readDir is an AsyncIterable and readDirSync an Iterable. Matching the
  // real shapes matters: `deno check` uses Deno's own type declarations, so a
  // shim that returns something else fails to typecheck every script that uses
  // it correctly.
  const entries = (p) => api.fs.list(p).map((e) => ({
    name: e.name, isDirectory: e.dir, isFile: !e.dir, isSymlink: false,
  }));
  D.readDir = (p) => {
    const items = entries(p);
    return { async *[Symbol.asyncIterator]() { for (const item of items) yield item; } };
  };
  D.readDirSync = (p) => entries(p);
  D.cwd = () => api.sys("cwd");
  D.env = {
    get: (n) => { const v = api.env(n); return v === "" ? undefined : v; },
    has: (n) => api.env(n) !== "",
  };
  // Deno.Command, minus the byte-array plumbing: stdout and stderr come back
  // as strings, which is what a script actually wants here.
  D.Command = class {
    constructor(program, options = {}) {
      this._program = program;
      this._args = options.args ?? [];
      this._cwd = options.cwd ?? ".";
    }
    outputSync() { return api.exec(this._program, this._args, this._cwd); }
    output() { return Promise.resolve(this.outputSync()); }
  };
  globalThis.fetch = (url, init = {}) => {
    const r = api.fetch(url, { method: init.method, body: init.body });
    return Promise.resolve({
      status: r.status,
      ok: r.status >= 200 && r.status < 300,
      text: () => Promise.resolve(r.body),
      json: () => Promise.resolve(JSON.parse(r.body)),
    });
  };

  globalThis.__bitty_deliver = async (mail) => {
    if (!handler) {
      api.log("script received mail but never called bitty.onMail(...)");
      if (mail.replyTo) ops.op_bitty_reply(mail.replyTo, "");
      return;
    }
    // Whatever the handler returns is the answer to a synchronous call. This
    // is what makes a script usable as a function rather than only as a peer.
    let result;
    try {
      result = await handler(mail, api);
    } catch (e) {
      if (mail.replyTo) ops.op_bitty_reply_error(mail.replyTo, String(e && e.message ? e.message : e));
      throw e;
    }
    if (mail.replyTo) {
      ops.op_bitty_reply(mail.replyTo, result === undefined || result === null ? "" : String(result));
    }
  };
  globalThis.__bitty_init = (info) => { Object.assign(api, info); };
  return api;
})();
"#;

pub async fn run(
    sys: Arc<System>,
    me: Meta,
    mailbox: UnboundedReceiver<Mail>,
    control: UnboundedReceiver<Control>,
    instructions: String,
    source: String,
) {
    // V8 is thread-bound; give this process its own thread and runtime, then
    // block on the actor loop there.
    let tag = me.tag.clone();
    let handle = std::thread::Builder::new()
        .name(me.id.clone())
        .spawn(move || {
            let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
                Ok(rt) => rt,
                Err(e) => {
                    ui::warn(&me.tag, &format!("could not start script runtime: {e}"));
                    return;
                }
            };
            rt.block_on(actor_loop(sys, me, mailbox, control, instructions, source));
        });
    if let Err(e) = handle {
        ui::warn(&tag, &format!("could not spawn script thread: {e}"));
    }
}

async fn actor_loop(
    sys: Arc<System>,
    me: Meta,
    mut mailbox: UnboundedReceiver<Mail>,
    mut control: UnboundedReceiver<Control>,
    instructions: String,
    source: String,
) {
    // Held as an Option so the old isolate can be dropped *before* the
    // replacement is built. Two isolates alive at once on the same thread
    // leaves V8 without a current handle scope and aborts the process.
    let mut runtime = match boot(&sys, &me, &instructions, &source).await {
        Some(runtime) => Some(runtime),
        None => return,
    };

    loop {
        me.set_status(Status::Idle);
        sys.note_quiesced();

        let mail = tokio::select! {
            // Code replacement is out of band, so it is never mistaken for a
            // message and cannot be starved behind a full mailbox.
            ctl = control.recv() => {
                match ctl {
                    Some(Control::Replace(source)) => {
                        ui::trace(&me.tag, "⟳ replacing script code");
                        drop(runtime.take());
                        match boot(&sys, &me, &instructions, &source).await {
                            Some(fresh) => runtime = Some(fresh),
                            None => return,
                        }
                        continue;
                    }
                    None => return,
                }
            }
            mail = mailbox.recv() => {
                // The sender is dropped when this process is stopped, which is
                // how a blocked script learns to exit.
                match mail { Some(mail) => mail, None => return }
            }
        };

        me.set_status(Status::Running);
        sys.note_running();

        let payload = json!({
            "from": mail.from,
            "fromName": mail.from_name,
            "body": mail.body,
            "priority": match mail.priority { Priority::Low => "low", Priority::High => "high" },
            "replyTo": mail.reply_to,
        });
        ui::trace(&me.tag, &format!("⇠ mail from {}", mail.from));
        sys.note_consumed(&me.id, mail.seq);

        let call = format!("globalThis.__bitty_deliver({payload});");
        let js = runtime.as_mut().expect("runtime is present while the loop runs");
        let outcome = match js.execute_script("[bitty:mail]", call) {
            Ok(_) => js
                .run_event_loop(PollEventLoopOptions::default())
                .await
                .map_err(|e| e.to_string()),
            Err(e) => Err(e.to_string()),
        };
        if let Err(e) = outcome {
            ui::warn(&me.tag, &format!("handler failed: {e}"));
            // A caller blocked on this message must not wait for the timeout
            // when the handler has already blown up.
            if let Some(id) = &mail.reply_to {
                if sys.call_is_pending(id) {
                    sys.resolve_call(id, Err(format!("the handler raised: {e}")));
                }
            }
        }
    }
}

/// Run TypeScript once, inline, with the calling process's own capabilities —
/// no id, no mailbox, no journal entry, no actor. Spawning a whole process to
/// evaluate an expression or reshape some JSON is a lot of machinery for a
/// computation that ends immediately.
///
/// It runs as the caller rather than as an attenuated child: this is the
/// caller acting, not delegating, so it can reach exactly what the caller can
/// and nothing more.
pub async fn run_inline(sys: Arc<System>, me: Meta, source: String, seconds: u64) -> (String, bool) {
    if let Err(e) = precheck_as("inline", &source, true) {
        return (e, true);
    }
    let (id, rx) = sys.register_call(&me.id, &me.id);
    let wrapped = format!(
        // The result is handed back through the same reply channel a script
        // process uses, so there is no second mechanism for getting a value
        // out of V8.
        "(async () => {{\n  const __value = await (async () => {{\n{source}\n}})();\n           Deno.core.ops.op_bitty_reply({id:?}, __value === undefined || __value === null ? \"\"          : String(__value));\n}})().catch((e) => Deno.core.ops.op_bitty_reply_error({id:?}, String(e && e.message ? e.message : e)));",
        id = id
    );

    let (sys_t, me_t, tag) = (sys.clone(), me.clone(), me.tag.clone());
    let started = std::thread::Builder::new().name(format!("{}-inline", me.id)).spawn(move || {
        let Ok(rt) = tokio::runtime::Builder::new_current_thread().enable_all().build() else {
            return;
        };
        rt.block_on(async move {
            let Some(mut runtime) = boot(&sys_t, &me_t, "", "").await else {
                return;
            };
            if let Err(e) = runtime.execute_script("[inline]", wrapped) {
                sys_t.resolve_call(&id, Err(format!("script error: {e}")));
                return;
            }
            let _ = runtime.run_event_loop(PollEventLoopOptions::default()).await;
        });
    });
    if started.is_err() {
        return ("could not start a runtime for the script".into(), true);
    }

    match tokio::time::timeout(std::time::Duration::from_secs(seconds), rx).await {
        Ok(Ok(Ok(value))) => (value, false),
        Ok(Ok(Err(e))) => (e, true),
        Ok(Err(_)) => ("the script ended without producing a value".into(), true),
        Err(_) => {
            ui::warn(&tag, "inline script timed out");
            ("the script did not finish in time".into(), true)
        }
    }
}

/// Build a fresh isolate for `source`. Returns None if the code cannot start,
/// having already reported why.
async fn boot(sys: &Arc<System>, me: &Meta, instructions: &str, source: &str) -> Option<JsRuntime> {
    const OPS: &[OpDecl] = &[
        op_bitty_send(),
        op_bitty_stop(),
        op_bitty_list(),
        op_bitty_reply(),
        op_bitty_reply_error(),
        op_bitty_log(),
        op_bitty_fs_read(),
        op_bitty_fs_write(),
        op_bitty_fs_list(),
        op_bitty_fs_mkdir(),
        op_bitty_fs_remove(),
        op_bitty_exec(),
        op_bitty_fetch(),
        op_bitty_env(),
        op_bitty_sys(),
    ];

    let host = Host {
        sys: sys.clone(),
        me: me.clone(),
    };
    let ext = Extension {
        name: "bitty",
        ops: std::borrow::Cow::Borrowed(OPS),
        op_state_fn: Some(Box::new(move |state: &mut OpState| {
            state.put(host);
        })),
        ..Default::default()
    };

    let mut runtime = JsRuntime::new(RuntimeOptions {
        extensions: vec![ext],
        ..Default::default()
    });

    // TypeScript is stripped to JavaScript before it ever reaches V8; deno_core
    // runs JS, not TS.
    let js = match transpile(&me.id, source) {
        Ok(js) => js,
        Err(e) => {
            ui::warn(&me.tag, &format!("TypeScript error: {e}"));
            finish(sys, me, "failed to compile");
            return None;
        }
    };

    if let Err(e) = runtime.execute_script("[bitty:prelude]", PRELUDE) {
        ui::warn(&me.tag, &format!("prelude failed: {e}"));
        finish(sys, me, "runtime failed to start");
        return None;
    }
    let info = json!({
        "id": me.id,
        "name": me.name,
        "parent": me.parent,
        "instructions": instructions,
    });
    let init = format!("globalThis.__bitty_init({info});");
    let _ = runtime.execute_script("[bitty:init]", init);

    if let Err(e) = runtime.execute_script("[script]", js) {
        ui::warn(&me.tag, &format!("script error: {e}"));
        finish(sys, me, "script raised at load");
        return None;
    }
    // Top-level awaits and pending promises from load.
    let _ = runtime.run_event_loop(PollEventLoopOptions::default()).await;
    Some(runtime)
}

/// A script that ends on its own is a normal exit; its links are told.
fn finish(sys: &Arc<System>, me: &Meta, reason: &str) {
    me.set_status(Status::Stopped);
    sys.signal_stalled(&me.id, &format!("{} — {reason}", me.label()));
}

/// The API a script is written against. Emitted next to the source when
/// typechecking so `bitty.onMail`, `api.fs.read` and the rest are known
/// symbols rather than errors — a typecheck against an undeclared API would
/// report nothing but noise.
const TYPES: &str = r#"
interface BittyMail { from: string; fromName: string | null; body: string;
  priority: "high" | "low"; replyTo: string | null; }
interface BittyExec { code: number | null; stdout: string; stderr: string; }
interface BittyEntry { name: string; path: string; dir: boolean; }
interface BittyApi {
  id: string; name: string | null; parent: string; instructions: string;
  onMail(handler: (mail: BittyMail, api: BittyApi) => unknown): void;
  send(to: string | string[], message: string, priority?: "high" | "low"): string;
  stop(targets: string | string[], cascade?: boolean): string;
  list(): string;
  log(text: string): void;
  fs: {
    read(path: string): string;
    write(path: string, contents: string): string;
    list(path: string): BittyEntry[];
    mkdir(path: string): string;
    remove(path: string): string;
  };
  exec(program: string, args?: string[], cwd?: string): BittyExec;
  fetch(url: string, opts?: { method?: string; body?: string }): { status: number; body: string };
  env(name: string): string;
  sys(key: string): string;
}
declare const bitty: BittyApi;
"#;

/// Validate a script before anything is spawned, so a mistake is reported to
/// whoever wrote it instead of killing a process that already claimed an id.
///
/// Syntax is always checked. Types are checked too when the `deno` binary is
/// available — it is the real compiler, so the diagnostics match what the
/// author would see locally. Without it we degrade to syntax only rather than
/// pretending nothing is wrong.
pub fn precheck(name: &str, source: &str) -> Result<(), String> {
    precheck_as(name, source, false)
}

/// `inline` wraps the source in a function body before checking, because
/// inline scripts legally use a top-level `return` — they run inside a wrapper
/// at execution time, so checking the bare text reports errors that are not
/// real.
pub fn precheck_as(name: &str, source: &str, inline: bool) -> Result<(), String> {
    let checked = if inline {
        format!("async function __inline(): Promise<unknown> {{\n{source}\n}}\nvoid __inline;")
    } else {
        source.to_string()
    };
    let source = checked.as_str();
    transpile(name, source).map_err(|e| format!("TypeScript syntax error: {e}"))?;

    let Ok(dir) = std::env::temp_dir().join(format!("bitty-check-{name}")).canonicalize().or_else(
        |_| {
            let dir = std::env::temp_dir().join(format!("bitty-check-{name}"));
            std::fs::create_dir_all(&dir).map(|_| dir)
        },
    ) else {
        return Ok(());
    };
    let types = dir.join("bitty.d.ts");
    let file = dir.join("script.ts");
    if std::fs::write(&types, TYPES).is_err()
        || std::fs::write(
            &file,
            format!("/// <reference path=\"./bitty.d.ts\" />\n{source}"),
        )
        .is_err()
    {
        return Ok(());
    }

    let output = std::process::Command::new("deno")
        .args(["check", "--no-lock", "--quiet"])
        .arg(&file)
        .output();
    let _ = std::fs::remove_dir_all(&dir);
    match output {
        // No deno binary: syntax has already been checked, which is the most
        // we can honestly promise here.
        Err(_) => Ok(()),
        Ok(output) if output.status.success() => Ok(()),
        Ok(output) => {
            let report = String::from_utf8_lossy(&output.stderr);
            // deno colors its diagnostics; the escapes are noise in a tool
            // result and worse in a terminal.
            let plain = strip_ansi(report.trim());
            Err(format!("TypeScript errors — fix these:\n{plain}"))
        }
    }
}

fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            for c in chars.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Strip TypeScript types. Deno's own transpiler, so the accepted syntax
/// matches what a `deno run` of the same file would accept.
fn transpile(name: &str, source: &str) -> anyhow::Result<String> {
    use deno_ast::{MediaType, ParseParams, SourceMapOption};
    let specifier = deno_ast::ModuleSpecifier::parse(&format!("file:///{name}.ts"))?;
    let parsed = deno_ast::parse_module(ParseParams {
        specifier,
        text: source.into(),
        media_type: MediaType::TypeScript,
        capture_tokens: false,
        scope_analysis: false,
        maybe_syntax: None,
    })?;
    let transpiled = parsed.transpile(
        &deno_ast::TranspileOptions::default(),
        &deno_ast::TranspileModuleOptions::default(),
        &deno_ast::EmitOptions {
            source_map: SourceMapOption::None,
            ..Default::default()
        },
    )?;
    Ok(transpiled.into_source().text)
}
