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
use std::cell::RefCell;
use std::rc::Rc;
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

/// Create processes from JSON specs — the same shape `spawn_topology` takes.
#[op2]
#[string]
fn op_bitty_spawn(
    state: &mut OpState,
    #[string] specs: String,
) -> Result<String, deno_error::JsErrorBox> {
    let nodes: Vec<Value> = match serde_json::from_str(&specs) {
        Ok(Value::Array(nodes)) => nodes,
        Ok(single) => vec![single],
        Err(e) => return Err(fs_error(format!("spawn expects a process spec: {e}"))),
    };
    host_call(state, |sys, me| actions::spawn(sys, me, &nodes))
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
    let resolved = host
        .me
        .grants
        .read
        .resolve(&path, false)
        .map_err(fs_error)?;
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
    let resolved = host
        .me
        .grants
        .write
        .resolve(&path, true)
        .map_err(fs_error)?;
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
    let resolved = host
        .me
        .grants
        .read
        .resolve(&path, false)
        .map_err(fs_error)?;
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
    let resolved = host
        .me
        .grants
        .write
        .resolve(&path, true)
        .map_err(fs_error)?;
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
    let resolved = host
        .me
        .grants
        .write
        .resolve(&path, false)
        .map_err(fs_error)?;
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
/// The checks and the spawn itself, shared by the two shapes a caller can ask
/// the result in: text for `api.exec`, raw bytes for `Deno.Command`.
fn run_program(
    state: &mut OpState,
    program: &str,
    args: &[String],
    cwd: &str,
) -> Result<std::process::Output, deno_error::JsErrorBox> {
    let host = state.borrow::<Host>();
    if !host.me.may(crate::grants::Capability::Run, program) {
        return Err(fs_error(format!(
            "not permitted to run '{program}'; you may run {}",
            host.me.permitted(crate::grants::Capability::Run)
        )));
    }
    // The working directory has to be somewhere this process can already read,
    // so running a command cannot become a way to reach outside its roots.
    let dir = host.me.grants.read.resolve(cwd, false).map_err(fs_error)?;
    std::process::Command::new(program)
        .args(args)
        .current_dir(&dir)
        .output()
        .map_err(|e| fs_error(format!("{program}: {e}")))
}

/// `api.exec`: output as text, which is what a script handling a command's
/// result almost always wants. Lossy for output that is not valid UTF-8, by
/// construction — a caller who needs the bytes themselves wants the op below.
#[op2]
#[string]
fn op_bitty_exec(
    state: &mut OpState,
    #[string] program: String,
    #[serde] args: Vec<String>,
    #[string] cwd: String,
) -> Result<String, deno_error::JsErrorBox> {
    let output = run_program(state, &program, &args, &cwd)?;
    Ok(json!({
        "code": output.status.code(),
        "stdout": String::from_utf8_lossy(&output.stdout),
        "stderr": String::from_utf8_lossy(&output.stderr),
    })
    .to_string())
}

/// What `Deno.Command` is specified to hand back: the bytes the program wrote,
/// as `Uint8Array`s. Routing those through a `String` first would be wrong
/// twice over — anything not valid UTF-8 would already have become replacement
/// characters, and the caller's `new TextDecoder().decode(out.stdout)` would be
/// handed a string where a buffer belongs. `ToJsBuffer` moves the vector into
/// V8 as a byte view, so nothing is copied through a text encoding at all.
#[derive(serde::Serialize)]
struct ExecBytes {
    /// None when the child was killed by a signal rather than exiting.
    code: Option<i32>,
    stdout: deno_core::ToJsBuffer,
    stderr: deno_core::ToJsBuffer,
}

#[op2]
#[serde]
fn op_bitty_exec_bytes(
    state: &mut OpState,
    #[string] program: String,
    #[serde] args: Vec<String>,
    #[string] cwd: String,
) -> Result<ExecBytes, deno_error::JsErrorBox> {
    let output = run_program(state, &program, &args, &cwd)?;
    Ok(ExecBytes {
        code: output.status.code(),
        stdout: output.stdout.into(),
        stderr: output.stderr.into(),
    })
}

/// Bind a port and serve it. The accept loop runs on the *main* runtime, not
/// in this isolate: a script process is mail-driven, and its event loop is only
/// pumped while a message is being handled, so a JS-side accept loop would
/// either block the mailbox or never be polled.
///
/// Instead a request becomes what everything else in this system is — a piece
/// of mail with a reply_to. It queues behind other messages, the handler runs
/// one at a time the way an actor should, and the reply plumbing is the same
/// plumbing `call_process` already uses.
#[op2(fast)]
fn op_bitty_serve(
    state: &mut OpState,
    #[string] hostname: String,
    #[smi] port: u32,
) -> Result<(), deno_error::JsErrorBox> {
    let host = state.borrow::<Host>();
    let with_port = format!("{hostname}:{port}");
    if !host.me.may(crate::grants::Capability::Net, &hostname)
        && !host.me.may(crate::grants::Capability::Net, &with_port)
    {
        return Err(fs_error(format!(
            "not permitted to serve on '{with_port}'; your network grant covers {}",
            host.me.permitted(crate::grants::Capability::Net)
        )));
    }

    let (sys, me) = (host.sys.clone(), host.me.clone());
    let listener = std::net::TcpListener::bind(&with_port)
        .map_err(|e| fs_error(format!("cannot bind {with_port}: {e}")))?;
    listener
        .set_nonblocking(true)
        .map_err(|e| fs_error(format!("{with_port}: {e}")))?;

    let tag = me.tag.clone();
    host.sys.rt().spawn(async move {
        let Ok(listener) = tokio::net::TcpListener::from_std(listener) else {
            ui::warn(&tag, "could not take over the listening socket");
            return;
        };
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                continue;
            };
            // One task per connection so a slow handler cannot wedge the
            // accept loop; the handlers themselves still serialize, because
            // they run in the script's single isolate.
            let (sys, me) = (sys.clone(), me.clone());
            tokio::spawn(async move {
                if let Err(e) = serve_connection(stream, &sys, &me).await {
                    ui::trace(&me.tag, &format!("  http: {e}"));
                }
            });
        }
    });
    Ok(())
}

/// Read one HTTP/1.1 request, hand it to the script as mail, write what comes
/// back. Deliberately minimal: one request per connection, closed afterward, so
/// there is no keep-alive state machine to get wrong.
async fn serve_connection(
    mut stream: tokio::net::TcpStream,
    sys: &Arc<System>,
    me: &Meta,
) -> Result<(), String> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    // Headers first: read until the blank line that ends them.
    let head_end = loop {
        if let Some(at) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break at + 4;
        }
        let n = stream.read(&mut chunk).await.map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("connection closed before the request finished".into());
        }
        buf.extend_from_slice(&chunk[..n]);
        if buf.len() > 1_000_000 {
            return Err("request headers too large".into());
        }
    };

    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let mut lines = head.lines();
    let request_line = lines.next().unwrap_or_default().to_string();
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("GET").to_string();
    let path = parts.next().unwrap_or("/").to_string();

    let mut headers = serde_json::Map::new();
    let mut content_length = 0usize;
    let mut authority = String::from("localhost");
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let (name, value) = (name.trim().to_lowercase(), value.trim().to_string());
        if name == "content-length" {
            content_length = value.parse().unwrap_or(0);
        }
        if name == "host" {
            authority = value.clone();
        }
        headers.insert(name, json!(value));
    }

    // Then the body, however much the headers said to expect.
    let mut body = buf[head_end..].to_vec();
    while body.len() < content_length {
        let n = stream.read(&mut chunk).await.map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..n]);
    }

    let (call, rx) = sys.register_call(&me.id, &me.id);
    let payload = json!({
        "__http": true,
        "method": method,
        "url": format!("http://{authority}{path}"),
        "headers": headers,
        "body": String::from_utf8_lossy(&body).to_string(),
    });
    if let Err(e) = sys.send(
        &me.id,
        Mail {
            from: "http".into(),
            from_name: Some("http".into()),
            body: payload.to_string(),
            priority: Priority::High,
            artifact_id: None,
            artifact_chars: None,
            reply_to: Some(call.clone()),
            seq: 0,
        },
    ) {
        sys.resolve_call(&call, Err(e.clone()));
        return Err(e);
    }

    // A handler that never answers must not hold the socket forever.
    let answered = tokio::time::timeout(std::time::Duration::from_secs(60), rx).await;
    let (status, extra, text) = match answered {
        Ok(Ok(Ok(value))) => match serde_json::from_str::<Value>(&value) {
            Ok(v) if v.get("__status").is_some() => (
                v["__status"].as_u64().unwrap_or(200) as u16,
                v["__headers"].clone(),
                v["__body"].as_str().unwrap_or("").to_string(),
            ),
            // A handler that returned a plain value rather than a Response.
            _ => (200, Value::Null, value),
        },
        Ok(Ok(Err(e))) => (500, Value::Null, e),
        _ => {
            sys.resolve_call(&call, Err("no response".into()));
            (504, Value::Null, "the handler did not respond".to_string())
        }
    };

    let mut head = format!(
        "HTTP/1.1 {status} {}\r\ncontent-length: {}\r\nconnection: close\r\n",
        reason(status),
        text.len()
    );
    let mut had_type = false;
    if let Some(map) = extra.as_object() {
        for (name, value) in map {
            let lower = name.to_lowercase();
            if lower == "content-length" || lower == "connection" {
                continue;
            }
            had_type |= lower == "content-type";
            head.push_str(&format!("{lower}: {}\r\n", value.as_str().unwrap_or("")));
        }
    }
    if !had_type {
        head.push_str("content-type: text/plain; charset=utf-8\r\n");
    }
    head.push_str("\r\n");
    stream
        .write_all(head.as_bytes())
        .await
        .map_err(|e| e.to_string())?;
    stream
        .write_all(text.as_bytes())
        .await
        .map_err(|e| e.to_string())?;
    stream.flush().await.map_err(|e| e.to_string())?;
    Ok(())
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        504 => "Gateway Timeout",
        _ => "OK",
    }
}

/// Wait, without holding the thread. This is an *async* op, so it yields to
/// the event loop, and the loop that drives a script process races the event
/// loop against the mailbox — which means a sleeping script is still reachable
/// and a message still lands the moment it arrives.
#[op2(async(deferred), nofast)]
async fn op_bitty_sleep(millis: f64) {
    let millis = millis.clamp(0.0, 86_400_000.0) as u64;
    tokio::time::sleep(std::time::Duration::from_millis(millis)).await;
}

/// Open sockets, keyed per process. A script refers to one by number and never
/// holds a handle, so a socket cannot outlive the isolate that opened it.
#[derive(Default)]
struct Sockets {
    next: u32,
    open: std::collections::HashMap<u32, SocketHandle>,
}

struct SocketHandle {
    outgoing: tokio::sync::mpsc::UnboundedSender<String>,
    incoming: Arc<tokio::sync::Mutex<tokio::sync::mpsc::UnboundedReceiver<Result<String, String>>>>,
}

/// Connect a WebSocket. This is the primitive that makes a script reactive
/// rather than a poller: the socket is driven on the main runtime, and the
/// script awaits the next frame instead of asking repeatedly whether anything
/// has changed.
#[op2(async(deferred), nofast)]
#[smi]
async fn op_bitty_ws_connect(
    state: Rc<RefCell<OpState>>,
    #[string] url: String,
) -> Result<u32, deno_error::JsErrorBox> {
    let (sys, me) = {
        let state = state.borrow();
        let host = state.borrow::<Host>();
        (host.sys.clone(), host.me.clone())
    };

    let parsed = reqwest::Url::parse(&url).map_err(|e| fs_error(format!("{url}: {e}")))?;
    let Some(hostname) = parsed.host_str().map(String::from) else {
        return Err(fs_error(format!("{url} names no host")));
    };
    let with_port = parsed
        .port()
        .map(|p| format!("{hostname}:{p}"))
        .unwrap_or_else(|| hostname.clone());
    if !me.may(crate::grants::Capability::Net, &hostname)
        && !me.may(crate::grants::Capability::Net, &with_port)
    {
        return Err(fs_error(format!(
            "not permitted to reach '{hostname}'; you may reach {}",
            me.permitted(crate::grants::Capability::Net)
        )));
    }

    let (stream, _) = tokio_tungstenite::connect_async(&url)
        .await
        .map_err(|e| fs_error(format!("{url}: {e}")))?;

    let (to_socket, mut outbox) = tokio::sync::mpsc::unbounded_channel::<String>();
    let (inbox_tx, inbox_rx) = tokio::sync::mpsc::unbounded_channel::<Result<String, String>>();

    // The socket is pumped on the main runtime, not in the isolate: V8 is not
    // Send, and this way a script that is busy handling a message is not also
    // responsible for keeping a connection alive.
    sys.rt().spawn(async move {
        use futures_util::{SinkExt, StreamExt};
        let (mut write, mut read) = stream.split();
        loop {
            tokio::select! {
                outgoing = outbox.recv() => match outgoing {
                    Some(text) => {
                        if write.send(tokio_tungstenite::tungstenite::Message::Text(text.into())).await.is_err() {
                            let _ = inbox_tx.send(Err("the socket closed while sending".into()));
                            return;
                        }
                    }
                    None => {
                        let _ = write.close().await;
                        return;
                    }
                },
                incoming = read.next() => match incoming {
                    Some(Ok(tokio_tungstenite::tungstenite::Message::Text(text))) => {
                        if inbox_tx.send(Ok(text.to_string())).is_err() { return; }
                    }
                    Some(Ok(tokio_tungstenite::tungstenite::Message::Close(_))) | None => {
                        let _ = inbox_tx.send(Err("the socket closed".into()));
                        return;
                    }
                    Some(Err(e)) => {
                        let _ = inbox_tx.send(Err(e.to_string()));
                        return;
                    }
                    // Ping/pong and binary frames are handled or ignored by the
                    // library; a text protocol is all this exposes.
                    Some(Ok(_)) => {}
                },
            }
        }
    });

    let mut state = state.borrow_mut();
    let sockets = state.try_borrow_mut::<Sockets>().is_none();
    if sockets {
        state.put(Sockets::default());
    }
    let sockets = state.borrow_mut::<Sockets>();
    sockets.next += 1;
    let id = sockets.next;
    sockets.open.insert(
        id,
        SocketHandle {
            outgoing: to_socket,
            incoming: Arc::new(tokio::sync::Mutex::new(inbox_rx)),
        },
    );
    Ok(id)
}

#[op2(fast)]
fn op_bitty_ws_send(
    state: &mut OpState,
    #[smi] id: u32,
    #[string] text: String,
) -> Result<(), deno_error::JsErrorBox> {
    let sockets = state
        .try_borrow::<Sockets>()
        .ok_or_else(|| fs_error("no socket is open".into()))?;
    let socket = sockets
        .open
        .get(&id)
        .ok_or_else(|| fs_error(format!("socket {id} is not open")))?;
    socket
        .outgoing
        .send(text)
        .map_err(|_| fs_error(format!("socket {id} has closed")))
}

/// Await the next frame. Returns null when the socket has closed, so a script
/// can loop on it without a separate liveness check.
#[op2(async(deferred), nofast)]
#[string]
async fn op_bitty_ws_recv(
    state: Rc<RefCell<OpState>>,
    #[smi] id: u32,
) -> Result<Option<String>, deno_error::JsErrorBox> {
    let incoming = {
        let state = state.borrow();
        let sockets = state
            .try_borrow::<Sockets>()
            .ok_or_else(|| fs_error("no socket is open".into()))?;
        sockets
            .open
            .get(&id)
            .ok_or_else(|| fs_error(format!("socket {id} is not open")))?
            .incoming
            .clone()
    };
    let mut incoming = incoming.lock().await;
    match incoming.recv().await {
        Some(Ok(text)) => Ok(Some(text)),
        Some(Err(_)) | None => Ok(None),
    }
}

#[op2(fast)]
fn op_bitty_ws_close(state: &mut OpState, #[smi] id: u32) {
    if let Some(sockets) = state.try_borrow_mut::<Sockets>() {
        sockets.open.remove(&id);
    }
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
    #[string] headers: String,
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

    // Headers are the difference between "can make a request" and "can use an
    // API": without them there is no way to authenticate, and the failure looks
    // like a rejected credential rather than a missing one.
    let supplied: serde_json::Map<String, Value> =
        serde_json::from_str(&headers).unwrap_or_default();

    let client = reqwest::Client::new();
    let (tx, rx) = std::sync::mpsc::channel();
    host.sys.rt().spawn(async move {
        let mut request = match method.to_uppercase().as_str() {
            "POST" => client.post(parsed).body(body),
            "PUT" => client.put(parsed).body(body),
            "PATCH" => client.patch(parsed).body(body),
            "DELETE" => client.delete(parsed).body(body),
            _ => client.get(parsed),
        };
        for (name, value) in &supplied {
            if let Some(value) = value.as_str() {
                request = request.header(name.as_str(), value);
            }
        }
        let outcome = match request.send().await {
            Ok(response) => {
                let status = response.status().as_u16();
                // Returned because a caller that cannot see rate-limit headers
                // has to guess at pacing, and will guess wrong.
                let mut received = serde_json::Map::new();
                for (name, value) in response.headers() {
                    if let Ok(value) = value.to_str() {
                        received.insert(name.as_str().to_string(), json!(value));
                    }
                }
                match response.text().await {
                    Ok(text) => Ok((status, text, received)),
                    Err(e) => Err(e.to_string()),
                }
            }
            Err(e) => Err(e.to_string()),
        };
        let _ = tx.send(outcome);
    });
    match rx.recv() {
        Ok(Ok((status, text, received))) => Ok(json!({
            "status": status,
            "body": text,
            "headers": Value::Object(received),
        })
        .to_string()),
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

/// Every variable this process may read, as a JSON object. Without it, a
/// process that cannot find a variable has no way to tell "not set" from
/// "misremembered the name", and its only recourse is to shell out to `env`
/// and grep — which is both a worse answer and an alarming-looking one.
/// The grant does the filtering, so this can never show more than `env` for a
/// single name already would.
#[op2]
#[string]
fn op_bitty_env_list(state: &mut OpState) -> Result<String, deno_error::JsErrorBox> {
    let host = state.borrow::<Host>();
    let mut out = serde_json::Map::new();
    for (name, value) in std::env::vars() {
        if host.me.may(crate::grants::Capability::Env, &name) {
            out.insert(name, json!(value));
        }
    }
    Ok(Value::Object(out).to_string())
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

/// A parent-created tool, invoked from this process's JS as an async
/// function. The alias was resolved and authority-checked at spawn; here the
/// arguments are validated against its schema and the call is delivered to
/// the target, exactly as the old schema-tool path did — the interface moved
/// into the session, the policy did not.
#[op2(async(deferred), nofast)]
#[string]
async fn op_bitty_call(
    state: Rc<RefCell<OpState>>,
    #[string] name: String,
    #[string] args: String,
) -> Result<String, deno_error::JsErrorBox> {
    let (sys, me) = {
        let state = state.borrow();
        let host = state.borrow::<Host>();
        (host.sys.clone(), host.me.clone())
    };
    let Some(alias) = me.aliases.iter().find(|a| a.name == name).cloned() else {
        return Err(fs_error(format!("no tool named '{name}'")));
    };
    let parsed: Value = serde_json::from_str(&args).unwrap_or_else(|_| json!({}));
    if let Err(e) = crate::agent::validate(&alias.input_schema, &parsed) {
        return Err(fs_error(format!("invalid arguments for '{name}': {e}")));
    }
    let body = serde_json::to_string(&parsed).unwrap_or_else(|_| "{}".into());
    let (reply, is_error) = crate::agent::call(
        &sys,
        &me,
        &alias.target,
        &body,
        crate::agent::DEFAULT_CALL_TIMEOUT,
    )
    .await;
    if is_error {
        return Err(fs_error(reply));
    }
    Ok(reply)
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
  let stopHandler = null;
  const api = {
    onMail(fn) {
      handler = fn;
      // Startup may await before registering. Anything that arrived in the
      // meantime is delivered now rather than answered with silence.
      const waiting = pending.splice(0, pending.length);
      for (const mail of waiting) globalThis.__bitty_deliver(mail);
    },
    // Runs once, right before this process's runtime is torn down — on
    // stop_process and on patch_script replacing this code. Close sockets,
    // flush anything not already written down; there is no turn after this
    // one to do it in.
    onStop(fn) {
      stopHandler = fn;
    },
    send(to, message, priority = "high") {
      return ops.op_bitty_send(Array.isArray(to) ? to : [to], String(message), priority);
    },
    stop(targets, cascade = false) {
      return ops.op_bitty_stop(Array.isArray(targets) ? targets : [targets], !!cascade);
    },
    list() { return ops.op_bitty_list(); },
    // Create processes. Takes one spec or a list of them, the same shape
    // spawn_topology takes, and returns the new ids.
    spawn(specs) {
      return ops.op_bitty_spawn(JSON.stringify(specs)).split(",").filter(Boolean);
    },
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
      const headers = opts.headers ? JSON.stringify(opts.headers) : "{}";
      const body = opts.body === undefined || opts.body === null
        ? "" : (typeof opts.body === "string" ? opts.body : JSON.stringify(opts.body));
      return JSON.parse(ops.op_bitty_fetch(
        String(url), String(opts.method ?? "GET"), body, headers));
    },
    env(name) { return ops.op_bitty_env(String(name)); },
    // Names, not values: enough to find out what you actually have without
    // pulling a pile of secrets into a transcript.
    envNames() { return Object.keys(JSON.parse(ops.op_bitty_env_list())).sort(); },
    // Waiting without blocking: the process stays reachable while it sleeps,
    // because the loop that drives it races the event loop against the mailbox.
    sleep(ms) { return ops.op_bitty_sleep(Number(ms)); },
    // A socket you await rather than poll. connect resolves to a handle;
    // recv resolves to the next frame, or null once the socket closes.
    async connect(url) {
      const id = await ops.op_bitty_ws_connect(String(url));
      return {
        id,
        send: (text) => ops.op_bitty_ws_send(id, typeof text === "string" ? text : JSON.stringify(text)),
        recv: () => ops.op_bitty_ws_recv(id),
        close: () => ops.op_bitty_ws_close(id),
        async *[Symbol.asyncIterator]() {
          for (;;) {
            const frame = await ops.op_bitty_ws_recv(id);
            if (frame === null) return;
            yield frame;
          }
        },
      };
    },
    sys(key) { return ops.op_bitty_sys(String(key)); },
    exec(program, args = [], cwd = ".") {
      return JSON.parse(ops.op_bitty_exec(String(program), args.map(String), String(cwd)));
    },
    log(text) { ops.op_bitty_log(String(text)); },
  };
  // `deno_core` does not install Deno's console extension. Provide the common
  // surface explicitly and route it through the same structured log op as
  // `api.log`; a script must never acquire a raw terminal writer.
  const consoleValue = (value) => {
    if (typeof value === "string") return value;
    if (value instanceof Error) return value.stack || value.message || String(value);
    try {
      const json = JSON.stringify(value);
      return json === undefined ? String(value) : json;
    } catch (_) {
      return String(value);
    }
  };
  const consoleLog = (...values) => api.log(values.map(consoleValue).join(" "));
  globalThis.console = Object.freeze({
    log: consoleLog,
    info: consoleLog,
    debug: consoleLog,
    warn: consoleLog,
    error: consoleLog,
  });
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
    // Filtered by the grant, so this shows nothing a per-name read would not
    // already give. A process that cannot find a variable should be able to
    // look, rather than shelling out to `env` and grepping for likely names.
    toObject: () => JSON.parse(ops.op_bitty_env_list()),
  };


  // Deno.Command. stdout and stderr are Uint8Arrays, as the real API specifies:
  // the idiom every caller writes is `new TextDecoder().decode(out.stdout)`, and
  // handing that a string used to produce silent mojibake instead of text.
  // `api.exec` keeps its own contract and still returns them as strings.
  D.Command = class {
    constructor(program, options = {}) {
      this._program = program;
      this._args = options.args ?? [];
      this._cwd = options.cwd ?? ".";
    }
    outputSync() {
      const r = ops.op_bitty_exec_bytes(
        String(this._program), this._args.map(String), String(this._cwd));
      // code is null when the child was killed by a signal, which is not success.
      return { code: r.code, success: r.code === 0, signal: null, stdout: r.stdout, stderr: r.stderr };
    }
    output() { return Promise.resolve(this.outputSync()); }
  };



  // Declared by the TypeScript lib, implemented by an extension we do not
  // embed. Same trap as URL: without these, code that decodes a subprocess's
  // output typechecks cleanly and throws at runtime. UTF-8 only.
  class TextDecoder {
    constructor(label = "utf-8") { this.encoding = String(label).toLowerCase(); }
    decode(input) {
      if (input == null) return "";
      // A string is not a BufferSource, and letting one through was silent
      // corruption rather than an error: bytes[i] came out a one-character
      // string, every range test below coerced to NaN and failed, and four
      // input characters collapsed into one garbage code point — ASCII in,
      // CJK mojibake out. Throw, the way the spec does.
      let bytes;
      if (input instanceof ArrayBuffer) bytes = new Uint8Array(input);
      else if (ArrayBuffer.isView(input)) {
        bytes = new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
      } else {
        throw new TypeError(
          "TextDecoder.decode expects an ArrayBuffer or a view of one, not " + typeof input);
      }
      let out = "";
      for (let i = 0; i < bytes.length; ) {
        const b = bytes[i];
        if (b < 0x80) { out += String.fromCharCode(b); i += 1; }
        else if (b >= 0xc0 && b < 0xe0) {
          out += String.fromCharCode(((b & 0x1f) << 6) | (bytes[i + 1] & 0x3f)); i += 2;
        } else if (b >= 0xe0 && b < 0xf0) {
          out += String.fromCharCode(((b & 0x0f) << 12) | ((bytes[i + 1] & 0x3f) << 6) | (bytes[i + 2] & 0x3f));
          i += 3;
        } else {
          const cp = ((b & 0x07) << 18) | ((bytes[i + 1] & 0x3f) << 12) | ((bytes[i + 2] & 0x3f) << 6) | (bytes[i + 3] & 0x3f);
          out += String.fromCodePoint(cp); i += 4;
        }
      }
      return out;
    }
  }
  class TextEncoder {
    constructor() { this.encoding = "utf-8"; }
    encode(input = "") {
      const s = String(input);
      const out = [];
      for (const ch of s) {
        const cp = ch.codePointAt(0);
        if (cp < 0x80) out.push(cp);
        else if (cp < 0x800) out.push(0xc0 | (cp >> 6), 0x80 | (cp & 0x3f));
        else if (cp < 0x10000) out.push(0xe0 | (cp >> 12), 0x80 | ((cp >> 6) & 0x3f), 0x80 | (cp & 0x3f));
        else out.push(0xf0 | (cp >> 18), 0x80 | ((cp >> 12) & 0x3f), 0x80 | ((cp >> 6) & 0x3f), 0x80 | (cp & 0x3f));
      }
      return new Uint8Array(out);
    }
  }
  globalThis.TextDecoder = TextDecoder;
  globalThis.TextEncoder = TextEncoder;

  // Enough of Headers/Request/Response for a handler to be written the way it
  // would be anywhere else. deno_core ships none of the fetch API, so these are
  // ours; they are not the full spec, and they are not pretending to be.
  class Headers {
    constructor(init) {
      this._m = new Map();
      if (init) for (const [k, v] of (init[Symbol.iterator] ? init : Object.entries(init))) this.set(k, v);
    }
    get(k) { const v = this._m.get(String(k).toLowerCase()); return v === undefined ? null : v; }
    set(k, v) { this._m.set(String(k).toLowerCase(), String(v)); }
    has(k) { return this._m.has(String(k).toLowerCase()); }
    delete(k) { this._m.delete(String(k).toLowerCase()); }
    entries() { return this._m.entries(); }
    forEach(fn) { for (const [k, v] of this._m) fn(v, k, this); }
    [Symbol.iterator]() { return this._m.entries(); }
    toJSON() { return Object.fromEntries(this._m); }
  }
  class Request {
    constructor(url, init = {}) {
      this.url = url;
      this.method = (init.method || "GET").toUpperCase();
      this.headers = init.headers instanceof Headers ? init.headers : new Headers(init.headers);
      this._body = init.body == null ? "" : String(init.body);
    }
    async text() { return this._body; }
    async json() { return JSON.parse(this._body || "null"); }
  }
  class Response {
    constructor(body, init = {}) {
      this._body = body == null ? "" : String(body);
      this.status = init.status || 200;
      this.headers = init.headers instanceof Headers ? init.headers : new Headers(init.headers);
      this.ok = this.status >= 200 && this.status < 300;
    }
    static json(value, init = {}) {
      const r = new Response(JSON.stringify(value), init);
      if (!r.headers.has("content-type")) r.headers.set("content-type", "application/json");
      return r;
    }
    async text() { return this._body; }
    async json() { return JSON.parse(this._body || "null"); }
  }
  // URL is declared by the TypeScript lib but implemented by an extension we
  // do not embed, so without this a handler that parses its own path compiles
  // cleanly and then throws at runtime. Enough of it to route a request.
  class URLSearchParams {
    constructor(init = "") {
      this._p = [];
      const s = String(init).replace(/^\?/, "");
      if (s) for (const pair of s.split("&")) {
        if (!pair) continue;
        const i = pair.indexOf("=");
        const k = i < 0 ? pair : pair.slice(0, i);
        const v = i < 0 ? "" : pair.slice(i + 1);
        this._p.push([decodeURIComponent(k.replace(/\+/g, " ")), decodeURIComponent(v.replace(/\+/g, " "))]);
      }
    }
    get(k) { const f = this._p.find(([n]) => n === k); return f ? f[1] : null; }
    getAll(k) { return this._p.filter(([n]) => n === k).map(([, v]) => v); }
    has(k) { return this._p.some(([n]) => n === k); }
    entries() { return this._p[Symbol.iterator](); }
    [Symbol.iterator]() { return this._p[Symbol.iterator](); }
    toString() { return this._p.map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`).join("&"); }
  }
  class URL {
    constructor(input, base) {
      let href = String(input);
      if (base && !/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(href)) {
        const b = new URL(base);
        href = href.startsWith("/") ? `${b.protocol}//${b.host}${href}` : `${b.protocol}//${b.host}${b.pathname.replace(/[^/]*$/, "")}${href}`;
      }
      const m = /^([a-zA-Z][a-zA-Z0-9+.-]*:)\/\/([^/?#]*)([^?#]*)(\?[^#]*)?(#.*)?$/.exec(href);
      if (!m) throw new TypeError(`Invalid URL: ${href}`);
      this.protocol = m[1];
      this.host = m[2];
      const at = this.host.lastIndexOf("@");
      const hostport = at < 0 ? this.host : this.host.slice(at + 1);
      const colon = hostport.lastIndexOf(":");
      this.hostname = colon < 0 ? hostport : hostport.slice(0, colon);
      this.port = colon < 0 ? "" : hostport.slice(colon + 1);
      this.pathname = m[3] || "/";
      this.search = m[4] || "";
      this.hash = m[5] || "";
      this.searchParams = new URLSearchParams(this.search);
      this.origin = `${this.protocol}//${this.host}`;
      this.href = href;
    }
    toString() { return this.href; }
  }
  globalThis.URL = URL;
  globalThis.URLSearchParams = URLSearchParams;
  globalThis.Headers = Headers;
  globalThis.Request = Request;
  globalThis.Response = Response;

  let httpHandler = null;
  Deno.serve = (a, b) => {
    const opts = typeof a === "function" ? {} : (a || {});
    const fn = typeof a === "function" ? a : (b || opts.handler);
    if (typeof fn !== "function") throw new TypeError("Deno.serve needs a handler function");
    if (httpHandler) throw new Error("this process is already serving; one server per process");
    const hostname = opts.hostname || "127.0.0.1";
    const port = opts.port === undefined ? 8000 : opts.port;
    ops.op_bitty_serve(hostname, port);
    httpHandler = fn;
    const addr = { transport: "tcp", hostname, port };
    if (opts.onListen) opts.onListen(addr);
    else api.log(`listening on http://${hostname}:${port}/`);
    // The server lives as long as the process does; stopping the process is
    // how it shuts down, so `finished` never settles on its own.
    return { addr, finished: new Promise(() => {}) };
  };

  globalThis.fetch = (url, init = {}) => {
    // Headers may arrive as a plain object or as anything iterable of pairs.
    let headers = init.headers;
    if (headers && typeof headers[Symbol.iterator] === "function") {
      headers = Object.fromEntries(headers);
    }
    const r = api.fetch(url, { method: init.method, body: init.body, headers });
    return Promise.resolve({
      status: r.status,
      ok: r.status >= 200 && r.status < 300,
      headers: {
        get: (name) => r.headers[String(name).toLowerCase()] ?? null,
        has: (name) => String(name).toLowerCase() in r.headers,
        entries: () => Object.entries(r.headers)[Symbol.iterator](),
        [Symbol.iterator]: () => Object.entries(r.headers)[Symbol.iterator](),
      },
      text: () => Promise.resolve(r.body),
      json: () => Promise.resolve(JSON.parse(r.body)),
    });
  };

  const pending = [];
  globalThis.__bitty_deliver = async (mail) => {
    // An HTTP request arrives as mail so that it queues, serializes and replies
    // like everything else — but it is dispatched to the serve handler, not to
    // onMail, so a script can be both a server and a peer.
    if (mail.from === "http" && httpHandler) {
      const req = JSON.parse(mail.body);
      let out;
      try {
        // The real Request throws if a GET or HEAD is given a body, even an
        // empty one, so the field is omitted rather than passed as "".
        const init = { method: req.method, headers: req.headers };
        if (req.body && req.method !== "GET" && req.method !== "HEAD") init.body = req.body;
        const request = new Request(req.url, init);
        const response = await httpHandler(request, { remoteAddr: null });
        const r = response instanceof Response ? response : new Response(String(response ?? ""));
        out = { __status: r.status, __headers: Object.fromEntries(r.headers.entries()), __body: await r.text() };
      } catch (e) {
        api.log(`http handler raised: ${e && e.message ? e.message : e}`);
        out = { __status: 500, __headers: {}, __body: "handler error" };
      }
      if (mail.replyTo) ops.op_bitty_reply(mail.replyTo, JSON.stringify(out));
      return;
    }
    if (!handler) {
      // Not an error yet: the handler may still be a few awaits away.
      pending.push(mail);
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
  globalThis.__bitty_stop = async () => {
    if (!stopHandler) return;
    try {
      await stopHandler(api);
    } catch (e) {
      api.log(`cleanup handler raised: ${e && e.message ? e.message : e}`);
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
    resumed: bool,
) {
    // V8 is thread-bound; give this process its own thread and runtime, then
    // block on the actor loop there.
    let tag = me.tag.clone();
    let handle = std::thread::Builder::new()
        .name(me.id.clone())
        .spawn(move || {
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    ui::warn(&me.tag, &format!("could not start script runtime: {e}"));
                    return;
                }
            };
            rt.block_on(actor_loop(
                sys,
                me,
                mailbox,
                control,
                instructions,
                source,
                resumed,
            ));
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
    resumed: bool,
) {
    // Held as an Option so the old isolate can be dropped *before* the
    // replacement is built. Two isolates alive at once on the same thread
    // leaves V8 without a current handle scope and aborts the process.
    let mut runtime = match boot(&sys, &me, &instructions, &source, resumed, false).await {
        Some(runtime) => Some(runtime),
        None => return,
    };

    // Why the event loop is raced rather than awaited to completion: a script
    // holding a timer or an open socket has work that is *permanently* pending,
    // so awaiting it would wedge the mailbox forever. Racing it against the
    // mailbox means async work keeps making progress while the process is idle,
    // and a message still gets through the moment it arrives. That is what
    // makes a script reactive instead of only ever running when mailed.
    //
    // `settled` records that the isolate has nothing pending, so the loop can
    // wait quietly on the mailbox rather than re-polling a finished event loop.
    let mut settled = false;
    // Whoever is blocked on the message currently being handled, so an async
    // failure can be reported to them rather than to no one.
    let mut pending_reply: Option<String> = None;

    enum Wake {
        Ctl(Option<Control>),
        Mail(Option<Mail>),
        Quiet(Result<(), String>),
    }

    loop {
        if settled {
            me.set_status(Status::Idle);
            sys.note_quiesced();
        }

        let wake = {
            let js = runtime
                .as_mut()
                .expect("runtime is present while the loop runs");
            tokio::select! {
                // Code replacement is out of band, so it is never mistaken for a
                // message and cannot be starved behind a full mailbox.
                ctl = control.recv() => Wake::Ctl(ctl),
                // recv is cancel-safe, so losing this race never drops a message.
                mail = mailbox.recv() => Wake::Mail(mail),
                result = js.run_event_loop(PollEventLoopOptions::default()), if !settled => {
                    Wake::Quiet(result.map_err(|e| e.to_string()))
                }
            }
        };

        let mail = match wake {
            Wake::Quiet(result) => {
                // Nothing left pending: stop polling and wait for mail.
                settled = true;
                if let Err(e) = result {
                    ui::warn(&me.tag, &format!("handler failed: {e}"));
                    if let Some(id) = pending_reply.take()
                        && sys.call_is_pending(&id)
                    {
                        sys.resolve_call(&id, Err(format!("the handler raised: {e}")));
                    }
                }
                continue;
            }
            Wake::Ctl(Some(Control::Replace(source))) => {
                ui::trace(&me.tag, "⟳ replacing script code");
                // The old code's sockets and closures die with this runtime,
                // so its onStop is the last chance to close them cleanly.
                run_cleanup(
                    runtime
                        .as_mut()
                        .expect("runtime is present while the loop runs"),
                    &me.tag,
                )
                .await;
                drop(runtime.take());
                match boot(&sys, &me, &instructions, &source, true, false).await {
                    Some(fresh) => runtime = Some(fresh),
                    None => return,
                }
                settled = false;
                continue;
            }
            // The sender is dropped when this process is stopped, which is how
            // a blocked script learns to exit.
            Wake::Ctl(None) | Wake::Mail(None) => {
                run_cleanup(
                    runtime
                        .as_mut()
                        .expect("runtime is present while the loop runs"),
                    &me.tag,
                )
                .await;
                return;
            }
            Wake::Mail(Some(mail)) => mail,
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
        pending_reply = mail.reply_to.clone();
        let js = runtime
            .as_mut()
            .expect("runtime is present while the loop runs");
        // Only the synchronous part runs here. Whatever the handler leaves
        // pending is driven by the race at the top of the loop, which is what
        // lets a handler await a socket without deafening the process.
        if let Err(e) = js.execute_script("[bitty:mail]", call) {
            let e = e.to_string();
            ui::warn(&me.tag, &format!("handler failed: {e}"));
            // A caller blocked on this message must not wait for the timeout
            // when the handler has already blown up.
            if let Some(id) = pending_reply.take()
                && sys.call_is_pending(&id)
            {
                sys.resolve_call(&id, Err(format!("the handler raised: {e}")));
            }
        }
        // A script has no turns, so this is its turn boundary: the point at
        // which what it has consumed is worth making durable.
        sys.journal.flush(&me.id);
        // A turn has just ended, so the log is consistent and nothing is
        // mid-write. Checked here rather than on a timer for that reason.
        if sys.journal.should_compact(&me.id) {
            sys.journal.compact(&me.id);
        }
        settled = false;
    }
}

/// What a session eval leaves behind for the next one: `g` (an alias for
/// `globalThis`) persists for the process's whole life, and oversized results
/// are parked in `g.results` instead of being pasted into the caller's
/// context — the model slices them programmatically instead of paying tokens
/// to read them.
const SESSION_PRELUDE: &str = r#"
globalThis.g = globalThis;
g.results = {};
globalThis.__bitty_result_seq = 0;
globalThis.__bitty_render = (v) => {
  if (v === undefined || v === null) return "";
  let s;
  if (typeof v === "string") s = v;
  else { try { s = JSON.stringify(v); } catch { s = String(v); } }
  const cap = 8000;
  if (s.length <= cap) return s;
  const key = "r" + (++globalThis.__bitty_result_seq);
  g.results[key] = v;
  return "[large result stored as g.results." + key + " — " + s.length +
    " chars serialized. Preview:]\n" + s.slice(0, 2000) +
    "\n…[slice g.results." + key + " in a later run_script call to see more]";
};
globalThis.__bitty_session_names = () => {
  const skip = new Set(globalThis.__bitty_skip || []);
  const names = Object.getOwnPropertyNames(globalThis)
    .filter((n) => !skip.has(n) && n !== "__bitty_skip");
  return JSON.stringify({ names, results: Object.keys(g.results) });
};
"#;

/// A process's persistent evaluation session: one long-lived isolate on its
/// own thread, holding state across `run_script` calls for the process's
/// whole life. This is what makes `run_script` a REPL rather than a
/// calculator — variables written to `g.*` survive from turn to turn, so an
/// agent can park data outside its context window and page it in selectively.
///
/// It runs as the owner rather than as an attenuated child: this is the
/// process acting, not delegating, so it reaches exactly what the process can
/// and nothing more. Session state is deliberately ephemeral across harness
/// restarts, like a script process's heap: anything that must survive belongs
/// in a file.
pub struct Session {
    tx: tokio::sync::mpsc::UnboundedSender<EvalRequest>,
    /// The isolate's thread-safe handle, for cancelling an eval that never
    /// finishes without losing the session.
    isolate: Arc<std::sync::Mutex<Option<deno_core::v8::IsolateHandle>>>,
}

struct EvalRequest {
    source: String,
    call_id: String,
}

impl Session {
    /// Start the session thread. Evals fail individually if the runtime
    /// cannot start; the owning process is unaffected.
    pub fn open(sys: Arc<System>, me: Meta) -> Session {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<EvalRequest>();
        let isolate = Arc::new(std::sync::Mutex::new(None));
        let slot = isolate.clone();
        let started = std::thread::Builder::new()
            .name(format!("{}-session", me.id))
            .spawn(move || session_thread(sys, me, rx, slot));
        if let Err(e) = started {
            // tx's receiver lives on the failed thread; evals will report.
            eprintln!("could not start a session thread: {e}");
        }
        Session { tx, isolate }
    }

    /// Evaluate one script in the session. The result comes back through the
    /// same reply channel a script process uses, so there is no second
    /// mechanism for getting a value out of V8.
    pub async fn eval(
        &self,
        sys: &Arc<System>,
        me: &Meta,
        source: &str,
        seconds: u64,
    ) -> (String, bool) {
        if let Err(e) = precheck_as("inline", source, true) {
            return (e, true);
        }
        let (id, rx) = sys.register_call(&me.id, &me.id);
        if self
            .tx
            .send(EvalRequest {
                source: source.to_string(),
                call_id: id.clone(),
            })
            .is_err()
        {
            sys.resolve_call(&id, Err(String::new()));
            return ("the session runtime is not available".into(), true);
        }
        match tokio::time::timeout(std::time::Duration::from_secs(seconds), rx).await {
            Ok(Ok(Ok(value))) => (value, false),
            Ok(Ok(Err(e))) => (e, true),
            Ok(Err(_)) => ("the script ended without producing a value".into(), true),
            Err(_) => {
                // Cancel the runaway execution but keep the session: state
                // already written to g.* survives, only this eval dies.
                if let Some(handle) = self.isolate.lock().unwrap().as_ref() {
                    handle.terminate_execution();
                }
                sys.resolve_call(&id, Err("timed out".into()));
                ui::warn(
                    &me.tag,
                    "session eval timed out; the running code was terminated",
                );
                (
                    "the script did not finish in time and was terminated. The session and its \
                     g.* state survive. If the work is genuinely long-running, do it in a script \
                     process instead."
                        .into(),
                    true,
                )
            }
        }
    }

    /// What the session is holding: (global names created by evals, keys of
    /// g.results). For the post-compaction notice.
    pub async fn state(&self, sys: &Arc<System>, me: &Meta) -> Option<(Vec<String>, Vec<String>)> {
        let (report, is_err) = self
            .eval(sys, me, "return globalThis.__bitty_session_names()", 5)
            .await;
        if is_err {
            return None;
        }
        let parsed: Value = serde_json::from_str(&report).ok()?;
        let list = |key: &str| -> Vec<String> {
            parsed[key]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        };
        Some((list("names"), list("results")))
    }
}

/// The session thread: boot once, then serve eval requests forever, racing
/// the isolate's event loop so async work keeps settling between requests —
/// the same structure as a script process's actor loop, with evals in place
/// of mail.
fn session_thread(
    sys: Arc<System>,
    me: Meta,
    mut rx: UnboundedReceiver<EvalRequest>,
    slot: Arc<std::sync::Mutex<Option<deno_core::v8::IsolateHandle>>>,
) {
    let Ok(rt) = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    else {
        return;
    };
    rt.block_on(async move {
        let Some(mut runtime) = boot(&sys, &me, "", "", false, true).await else {
            while let Some(req) = rx.recv().await {
                sys.resolve_call(&req.call_id, Err("the session runtime failed to start".into()));
            }
            return;
        };
        if let Err(e) = runtime.execute_script("[bitty:session]", SESSION_PRELUDE) {
            ui::warn(&me.tag, &format!("session prelude failed: {e}"));
        }
        // Baseline the namespace after the prelude, so the state probe
        // reports only names evals created.
        let _ = runtime.execute_script(
            "[bitty:baseline]",
            "globalThis.__bitty_skip = Object.getOwnPropertyNames(globalThis);",
        );
        *slot.lock().unwrap() = Some(runtime.v8_isolate().thread_safe_handle());

        let mut settled = false;
        loop {
            enum Wake {
                Req(Option<EvalRequest>),
                Quiet(Result<(), String>),
            }
            let wake = tokio::select! {
                req = rx.recv() => Wake::Req(req),
                result = runtime.run_event_loop(PollEventLoopOptions::default()), if !settled => {
                    Wake::Quiet(result.map_err(|e| e.to_string()))
                }
            };
            match wake {
                Wake::Quiet(result) => {
                    settled = true;
                    if let Err(e) = result {
                        ui::warn(&me.tag, &format!("session async work failed: {e}"));
                    }
                }
                // The Session was dropped: its owner stopped.
                Wake::Req(None) => return,
                Wake::Req(Some(req)) => {
                    // A previous eval may have been terminated on timeout;
                    // clear the flag or this one dies with it.
                    runtime.v8_isolate().cancel_terminate_execution();
                    let wrapped = format!(
                        "(async () => {{\n  const __value = await (async () => {{\n{}\n}})();\n  \
                         Deno.core.ops.op_bitty_reply({id:?}, globalThis.__bitty_render(__value));\n\
                         }})().catch((e) => Deno.core.ops.op_bitty_reply_error({id:?}, \
                         String(e && e.message ? e.message : e)));",
                        req.source,
                        id = req.call_id
                    );
                    // Strip the types before V8 sees them. The precheck
                    // transpiled too, but only to validate — running the
                    // original source would hand V8 every annotation as a
                    // syntax error.
                    match transpile("inline", &wrapped) {
                        Ok(js) => {
                            if let Err(e) = runtime.execute_script("[inline]", js) {
                                sys.resolve_call(&req.call_id, Err(format!("script error: {e}")));
                            }
                        }
                        Err(e) => {
                            sys.resolve_call(&req.call_id, Err(format!("script error: {e}")));
                        }
                    }
                    settled = false;
                }
            }
        }
    });
}

/// Build a fresh isolate for `source`. Returns None if the code cannot start,
/// having already reported why.
///
/// `for_session` marks an isolate that belongs to an *agent's* session rather
/// than to a script process: a boot failure is then reported to the caller
/// and nothing else — marking the process stopped would tombstone a live
/// agent over a scripting hiccup.
async fn boot(
    sys: &Arc<System>,
    me: &Meta,
    instructions: &str,
    source: &str,
    resumed: bool,
    for_session: bool,
) -> Option<JsRuntime> {
    const OPS: &[OpDecl] = &[
        op_bitty_send(),
        op_bitty_stop(),
        op_bitty_list(),
        op_bitty_spawn(),
        op_bitty_reply(),
        op_bitty_reply_error(),
        op_bitty_log(),
        op_bitty_fs_read(),
        op_bitty_fs_write(),
        op_bitty_fs_list(),
        op_bitty_fs_mkdir(),
        op_bitty_fs_remove(),
        op_bitty_exec(),
        op_bitty_exec_bytes(),
        op_bitty_fetch(),
        op_bitty_serve(),
        op_bitty_sleep(),
        op_bitty_ws_connect(),
        op_bitty_ws_send(),
        op_bitty_ws_recv(),
        op_bitty_ws_close(),
        op_bitty_env(),
        op_bitty_env_list(),
        op_bitty_sys(),
        op_bitty_call(),
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
    // A script is executed as a classic script, where `await` outside a
    // function is a syntax error — but it is *typechecked* as a module, where
    // top-level await is perfectly legal. So source that compiles cleanly could
    // fail to parse, reported as "missing ) after argument list", which points
    // nowhere near the actual line. Wrapping in an async IIFE makes the two
    // agree: top-level await works, and the declarations stay visible to the
    // handler because it closes over this same scope.
    let wrapped = format!(
        "(async () => {{\n{source}\n}})().catch((e) => {{ \
         bitty.log(\"script failed during startup: \" + (e && e.message ? e.message : e)); }});"
    );
    let source = wrapped.as_str();
    let js = match transpile(&me.id, source) {
        Ok(js) => js,
        Err(e) => {
            ui::warn(&me.tag, &format!("TypeScript error: {e}"));
            if !for_session {
                finish(sys, me, "failed to compile");
            }
            return None;
        }
    };
    // Diagnostic escape hatch: dump exactly what V8 is asked to parse.
    if let Ok(path) = std::env::var("BITTY_DUMP_JS") {
        let _ = std::fs::write(path, &js);
    }

    if let Err(e) = runtime.execute_script("[bitty:prelude]", PRELUDE) {
        ui::warn(&me.tag, &format!("prelude failed: {e}"));
        if !for_session {
            finish(sys, me, "runtime failed to start");
        }
        return None;
    }
    let info = json!({
        "id": me.id,
        "name": me.name,
        "parent": me.parent,
        "instructions": instructions,
        // True when this is the harness coming back up rather than a first
        // start. The source re-runs either way — that is how a script gets its
        // sockets and handlers back — so this is how it tells "connect again"
        // apart from "do the one-time setup".
        "resumed": resumed,
    });
    let init = format!("globalThis.__bitty_init({info});");
    let _ = runtime.execute_script("[bitty:init]", init);

    // Parent-created tools become async functions in this namespace — a
    // script and an agent's session see them identically. The holder never
    // needs to know which process answers; the resolved target lives on the
    // Rust side of the op.
    if !me.aliases.is_empty() {
        let wrappers: String = me
            .aliases
            .iter()
            .map(|alias| {
                let name = serde_json::to_string(&alias.name).unwrap_or_default();
                format!(
                    "globalThis[{name}] = async (args = {{}}) => \
                     Deno.core.ops.op_bitty_call({name}, JSON.stringify(args));\n"
                )
            })
            .collect();
        if let Err(e) = runtime.execute_script("[bitty:tools]", wrappers) {
            ui::warn(&me.tag, &format!("tool injection failed: {e}"));
        }
    }

    if let Err(e) = runtime.execute_script("[script]", js) {
        ui::warn(&me.tag, &format!("script error: {e}"));
        if !for_session {
            finish(sys, me, "script raised at load");
        }
        return None;
    }
    // Give startup a moment to settle, but no more: a script whose top level
    // runs forever — connect, then read frames until the process is stopped —
    // would otherwise never return from boot, and a process that never leaves
    // boot never reaches its mailbox. It looks alive, connects, and answers
    // nothing. Whatever is still pending is driven by the actor loop's race.
    let _ = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        runtime.run_event_loop(PollEventLoopOptions::default()),
    )
    .await;
    Some(runtime)
}

/// Give a stopping script's registered `onStop` a chance to run — closing
/// sockets, flushing anything not already written down — before its runtime
/// is dropped. Bounded like boot's own startup settle above: a handler that
/// never resolves must not hang the stop it is trying to clean up after.
async fn run_cleanup(runtime: &mut JsRuntime, tag: &ui::Tag) {
    if let Err(e) = runtime.execute_script("[bitty:stop]", "globalThis.__bitty_stop();") {
        ui::warn(tag, &format!("cleanup handler failed: {e}"));
        return;
    }
    match tokio::time::timeout(
        std::time::Duration::from_secs(10),
        runtime.run_event_loop(PollEventLoopOptions::default()),
    )
    .await
    {
        Ok(Err(e)) => ui::warn(tag, &format!("cleanup handler failed: {e}")),
        Err(_) => ui::warn(tag, "cleanup handler timed out after 10s"),
        Ok(Ok(())) => {}
    }
}

/// A script that ends on its own is a normal exit; its links are told.
fn finish(sys: &Arc<System>, me: &Meta, reason: &str) {
    me.set_status(Status::Stopped);
    sys.signal_stalled(&me.id, &format!("{} — {reason}", me.label()));
}

/// Validate a script before anything is spawned, so a mistake is reported to
/// whoever wrote it instead of killing a process that already claimed an id.
///
/// Syntax only, via the same embedded transpiler that runs the script —
/// no subprocess, no temp files, no dependency on a `deno` binary being
/// installed on the host. A full type-check would need the TypeScript
/// compiler itself, which isn't embedded; shelling out to a host `deno`
/// for that traded a self-contained harness for one that silently picks up
/// whatever `deno` happens to be on PATH, which is exactly what broke when
/// it resolved a wrapper script belonging to an unrelated project.
pub fn precheck(name: &str, source: &str) -> Result<(), String> {
    precheck_as(name, source, false)
}

/// `inline` wraps the source in a function body before checking, because
/// inline scripts legally use a top-level `return` — they run inside a wrapper
/// at execution time, so checking the bare text reports errors that are not
/// real.
pub fn precheck_as(name: &str, source: &str, inline: bool) -> Result<(), String> {
    let checked = if inline {
        // The return type is inferred rather than declared: annotating it
        // Promise<unknown> makes TypeScript demand a return statement, which
        // rejects every inline script that only has side effects.
        format!("async function __inline() {{\n{source}\n}}\nvoid __inline;")
    } else {
        source.to_string()
    };
    transpile(name, &checked).map_err(|e| format!("TypeScript syntax error: {e}"))?;
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grants::{Grant, Grants, PathGrant};
    use std::collections::HashSet;
    use std::sync::atomic::AtomicU64;

    fn test_meta(id: &str, write_root: std::path::PathBuf) -> Meta {
        Meta {
            id: id.to_string(),
            name: None,
            parent: "user".to_string(),
            tag: ui::Tag::new(id, 0),
            status: Arc::new(std::sync::Mutex::new(Status::Running)),
            persona: None,
            grants: Grants {
                send: Grant::Nobody,
                stop: Grant::Ids(HashSet::from([id.to_string()])),
                spawn: Grant::Nobody,
                run: Grant::Nobody,
                net: Grant::Nobody,
                env: Grant::Nobody,
                sys: Grant::Nobody,
                read: PathGrant::Nowhere,
                write: PathGrant::Under(vec![write_root]),
            },
            labels: std::collections::HashMap::new(),
            context_tokens: Arc::new(AtomicU64::new(0)),
            aliases: Vec::new(),
            model: Arc::new(std::sync::Mutex::new("small".to_string())),
            effort: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    // Proves the onStop wiring end to end: JS registers a handler, run_cleanup
    // drives it via __bitty_stop, and the handler's own async fs write (the
    // same thing a real cleanup would do — close a socket, flush state) lands
    // on disk before run_cleanup returns.
    #[tokio::test]
    async fn on_stop_handler_runs_before_the_runtime_is_torn_down() {
        test_key();
        // Real grant roots are always canonicalized at spawn time (system.rs);
        // macOS's temp dir is itself a symlink (/var -> /private/var), so
        // skipping that here would reject every write as "outside" the root.
        let temp = std::env::temp_dir().canonicalize().unwrap();
        let marker = temp.join(format!("bitty-onstop-test-{}", std::process::id()));
        let _ = std::fs::remove_file(&marker);

        let client = crate::api::Client::from_env().unwrap();
        let sys = Arc::new(System::new(client));
        let me = test_meta("proc-test", temp);

        let source = format!(
            r#"
            bitty.onMail(async () => {{}});
            bitty.onStop(async () => {{
              await Deno.writeTextFile({marker:?}, "cleaned up");
            }});
            "#
        );

        let mut runtime = boot(&sys, &me, "test", &source, false, false)
            .await
            .expect("boot should succeed");
        run_cleanup(&mut runtime, &me.tag).await;

        let contents =
            std::fs::read_to_string(&marker).expect("onStop should have written the marker file");
        assert_eq!(contents, "cleaned up");
        let _ = std::fs::remove_file(&marker);
    }

    /// `Client::from_env` reads ANTHROPIC_API_KEY, and it never makes a network
    /// call — script processes have no access to sys.api at all. Every test
    /// here goes through this before building a client, so the single write is
    /// ordered ahead of every read of it.
    fn test_key() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        // SAFETY (test-only): behind a Once, so it happens exactly once and
        // strictly before any Client::from_env below; nothing ever unsets it.
        ONCE.call_once(|| unsafe {
            std::env::set_var("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
        });
    }

    /// Boot a throwaway script runtime that may run python3 inside the temp
    /// directory, run `body` as its top level, and hand back what the body
    /// returned. A test has no other channel into an isolate, so the value
    /// comes out through a marker file — the same trick the onStop test above
    /// uses. `TMP` is in scope for the body, as a directory to run in.
    async fn eval_json(body: &str) -> Value {
        test_key();
        // Real grant roots are always canonicalized at spawn time (system.rs);
        // macOS's temp dir is itself a symlink (/var -> /private/var), so
        // skipping that here would reject every path as "outside" the root.
        let temp = std::env::temp_dir().canonicalize().unwrap();
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let marker = temp.join(format!(
            "bitty-script-test-{}-{}.json",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let _ = std::fs::remove_file(&marker);

        let client = crate::api::Client::from_env().unwrap();
        let sys = Arc::new(System::new(client));
        let mut me = test_meta("proc-test", temp.clone());
        me.grants.run = Grant::Ids(HashSet::from(["python3".to_string()]));
        me.grants.read = PathGrant::Under(vec![temp.clone()]);

        let source = format!(
            r#"
            const TMP = {temp:?};
            try {{
              const __value = (() => {{ {body} }})();
              bitty.fs.write({marker:?}, JSON.stringify(__value));
            }} catch (e) {{
              bitty.fs.write({marker:?}, JSON.stringify({{
                threw: String(e && e.name),
                message: String(e && e.message ? e.message : e),
              }}));
            }}
            "#
        );
        let runtime = boot(&sys, &me, "test", &source, false, false).await;
        assert!(runtime.is_some(), "the test script should have loaded");
        let raw = std::fs::read_to_string(&marker)
            .expect("the test script should have written its result");
        let _ = std::fs::remove_file(&marker);
        serde_json::from_str(&raw).expect("the test script should have written JSON")
    }

    /// A `Deno.Command` that has python3 write exactly `payload` (a python bytes
    /// expression) to `stream`, so a test can ask for an odd length or for
    /// bytes that are not valid UTF-8 without a shell in the way.
    fn emit(stream: &str, payload: &str) -> String {
        format!(
            r#"const out = new Deno.Command("python3", {{
                 args: ["-c", "import sys; sys.{stream}.buffer.write({payload})"],
                 cwd: TMP,
               }}).outputSync();"#
        )
    }

    // REGRESSION (the bug this group exists for): `Deno.Command`'s output used
    // to come back as a JS *string*, and the TextDecoder polyfill accepted a
    // string silently — `bytes[i]` was a one-character string, every range test
    // coerced to NaN and failed, and four input characters collapsed into one
    // garbage code point. ASCII in, CJK mojibake out: `echo hello` decoded to
    // "\u{0}\u{0}", and digit-rich output (digits being the one character class
    // that coerces to a number) produced glyphs like "Ɂ ቅ ⁄ ㇃ 䃂 熃 耆". Nothing
    // was ever wrong with the bytes, only with JS's view of them. This pins the
    // whole round trip: what the program printed is what decode() returns.
    #[tokio::test]
    async fn subprocess_stdout_decodes_to_exactly_what_the_program_printed() {
        let expected = "Bitty 0123456789 the quick brown fox";
        let v = eval_json(&format!(
            "{} return {{ text: new TextDecoder().decode(out.stdout) }};",
            emit("stdout", &format!("b'{expected}'"))
        ))
        .await;
        assert_eq!(
            v["text"], expected,
            "stdout should decode verbatim, got {v}"
        );
    }

    /// The shape of the value, not only its contents: a two-byte element view
    /// is exactly what turns ASCII into CJK, so pin element width and length.
    #[tokio::test]
    async fn subprocess_stdout_is_a_uint8array_of_the_true_byte_length() {
        let v = eval_json(&format!(
            "{} return {{ ctor: out.stdout.constructor.name, bpe: out.stdout.BYTES_PER_ELEMENT, \
             len: out.stdout.length, byteLength: out.stdout.byteLength, \
             isView: ArrayBuffer.isView(out.stdout) }};",
            emit("stdout", "b'hello there'")
        ))
        .await;
        assert_eq!(
            v["ctor"], "Uint8Array",
            "stdout should be a Uint8Array, got {v}"
        );
        assert_eq!(
            v["bpe"], 1,
            "stdout elements should be single bytes, got {v}"
        );
        assert_eq!(
            v["len"], 11,
            "stdout length should be the byte count, got {v}"
        );
        assert_eq!(
            v["byteLength"], 11,
            "stdout byteLength should be the byte count, got {v}"
        );
        assert_eq!(
            v["isView"], true,
            "stdout should be a view over a buffer, got {v}"
        );
    }

    /// An odd byte count is unrepresentable in a two-byte-element view, so this
    /// fails loudly if the buffer is ever reinterpreted as UTF-16.
    #[tokio::test]
    async fn subprocess_stdout_survives_an_odd_byte_count() {
        let v = eval_json(&format!(
            "{} return {{ len: out.stdout.length, text: new TextDecoder().decode(out.stdout) }};",
            emit("stdout", "b'abcde'")
        ))
        .await;
        assert_eq!(
            v["len"], 5,
            "five bytes should arrive as five elements, got {v}"
        );
        assert_eq!(
            v["text"], "abcde",
            "an odd-length payload should decode intact, got {v}"
        );
    }

    /// Multi-byte UTF-8 must be decoded once and only once — a double decode or
    /// a lossy pass shows up here as mojibake or U+FFFD.
    #[tokio::test]
    async fn subprocess_stdout_decodes_multibyte_utf8() {
        let v = eval_json(&format!(
            "{} return {{ len: out.stdout.length, text: new TextDecoder().decode(out.stdout) }};",
            emit("stdout", r"'h\u00e9llo \u2713'.encode('utf-8')")
        ))
        .await;
        assert_eq!(
            v["len"], 10,
            "h\u{e9}llo \u{2713} is ten UTF-8 bytes, got {v}"
        );
        assert_eq!(
            v["text"], "h\u{e9}llo \u{2713}",
            "multibyte UTF-8 should survive, got {v}"
        );
    }

    /// Bytes that are not valid UTF-8 reach JS unchanged: the buffer is the
    /// program's output, not a string that was decoded on the way there.
    #[tokio::test]
    async fn subprocess_stdout_preserves_bytes_that_are_not_valid_utf8() {
        let v = eval_json(&format!(
            "{} return {{ bytes: Array.from(out.stdout) }};",
            emit("stdout", "bytes([0xff, 0xfe, 0x41])")
        ))
        .await;
        assert_eq!(
            v["bytes"],
            json!([255, 254, 65]),
            "invalid UTF-8 should survive, got {v}"
        );
    }

    /// stderr goes through the identical conversion, and it is the half a
    /// caller reaches for when a command fails.
    #[tokio::test]
    async fn subprocess_stderr_is_bytes_like_stdout() {
        let v = eval_json(&format!(
            "{} return {{ ctor: out.stderr.constructor.name, bpe: out.stderr.BYTES_PER_ELEMENT, \
             err: new TextDecoder().decode(out.stderr), outLen: out.stdout.length }};",
            emit("stderr", "b'boom 42'")
        ))
        .await;
        assert_eq!(
            v["ctor"], "Uint8Array",
            "stderr should be a Uint8Array, got {v}"
        );
        assert_eq!(
            v["bpe"], 1,
            "stderr elements should be single bytes, got {v}"
        );
        assert_eq!(
            v["err"], "boom 42",
            "stderr should decode verbatim, got {v}"
        );
        assert_eq!(v["outLen"], 0, "nothing was written to stdout, got {v}");
    }

    /// Nothing in the conversion may chunk or truncate: 100_000 bytes is well
    /// past any 64 KiB pipe buffer.
    #[tokio::test]
    async fn subprocess_stdout_survives_a_payload_larger_than_64_kib() {
        let v = eval_json(&format!(
            "{} const text = new TextDecoder().decode(out.stdout); \
             return {{ len: out.stdout.length, textLen: text.length, \
             matches: text === '0123456789'.repeat(10000) }};",
            emit("stdout", "b'0123456789' * 10000")
        ))
        .await;
        assert_eq!(v["len"], 100_000, "every byte should arrive, got {v}");
        assert_eq!(v["textLen"], 100_000, "every byte should decode, got {v}");
        assert_eq!(
            v["matches"], true,
            "a large payload should decode verbatim, got {v}"
        );
    }

    /// The rest of the documented CommandOutput shape, which the shim used to
    /// omit entirely: a caller checking `out.success` got undefined.
    #[tokio::test]
    async fn subprocess_output_reports_success_and_exit_code() {
        let v = eval_json(
            r#"const ok = new Deno.Command("python3", { args: ["-c", "pass"], cwd: TMP }).outputSync();
               const bad = new Deno.Command("python3", { args: ["-c", "raise SystemExit(3)"], cwd: TMP }).outputSync();
               return { okCode: ok.code, okSuccess: ok.success, badCode: bad.code, badSuccess: bad.success };"#,
        )
        .await;
        assert_eq!(v["okCode"], 0, "a clean exit is code 0, got {v}");
        assert_eq!(v["okSuccess"], true, "a clean exit is a success, got {v}");
        assert_eq!(
            v["badCode"], 3,
            "the child's exit code should come through, got {v}"
        );
        assert_eq!(
            v["badSuccess"], false,
            "a nonzero exit is not a success, got {v}"
        );
    }

    /// The failure mode that cost two processes an hour: decode() was handed
    /// something that is not a BufferSource and returned garbage rather than
    /// raising. A TypeError, as the spec requires, is the whole point.
    #[tokio::test]
    async fn text_decoder_rejects_a_string_instead_of_silently_decoding_it() {
        let v = eval_json(
            r#"try {
                 const bad = new TextDecoder().decode("already a string");
                 return { threw: null, returned: bad };
               } catch (e) { return { threw: String(e.name), message: String(e.message) }; }"#,
        )
        .await;
        assert_eq!(
            v["threw"], "TypeError",
            "decoding a string should throw, got {v}"
        );
    }

    /// The encoder and decoder are both ours, so round-tripping through them is
    /// the cheapest proof that neither drops a multibyte sequence.
    #[tokio::test]
    async fn text_encoder_and_decoder_round_trip_multibyte_utf8() {
        let v = eval_json(
            r#"const bytes = new TextEncoder().encode("h\u00e9llo \u2713");
               return { ctor: bytes.constructor.name, len: bytes.length,
                        back: new TextDecoder().decode(bytes) };"#,
        )
        .await;
        assert_eq!(
            v["ctor"], "Uint8Array",
            "encode should produce bytes, got {v}"
        );
        assert_eq!(
            v["len"], 10,
            "h\u{e9}llo \u{2713} is ten UTF-8 bytes, got {v}"
        );
        assert_eq!(
            v["back"], "h\u{e9}llo \u{2713}",
            "the round trip should be lossless, got {v}"
        );
    }

    /// A view into the middle of a buffer decodes its own window, not the whole
    /// allocation behind it.
    #[tokio::test]
    async fn text_decoder_honours_a_view_offset() {
        let v = eval_json(
            r#"const all = new Uint8Array([65, 66, 67, 68]);
               return { part: new TextDecoder().decode(all.subarray(1, 3)),
                        whole: new TextDecoder().decode(all),
                        buffer: new TextDecoder().decode(all.buffer) };"#,
        )
        .await;
        assert_eq!(
            v["part"], "BC",
            "a subarray should decode only its window, got {v}"
        );
        assert_eq!(
            v["whole"], "ABCD",
            "a full view should decode wholly, got {v}"
        );
        assert_eq!(
            v["buffer"], "ABCD",
            "an ArrayBuffer should still decode, got {v}"
        );
    }
}
