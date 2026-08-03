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
    stream.write_all(head.as_bytes()).await.map_err(|e| e.to_string())?;
    stream.write_all(text.as_bytes()).await.map_err(|e| e.to_string())?;
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
    let millis = millis.max(0.0).min(86_400_000.0) as u64;
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
    let supplied: serde_json::Map<String, Value> = serde_json::from_str(&headers)
        .unwrap_or_default();

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



  // Declared by the TypeScript lib, implemented by an extension we do not
  // embed. Same trap as URL: without these, code that decodes a subprocess's
  // output typechecks cleanly and throws at runtime. UTF-8 only.
  class TextDecoder {
    constructor(label = "utf-8") { this.encoding = String(label).toLowerCase(); }
    decode(input) {
      if (input == null) return "";
      const bytes = input instanceof ArrayBuffer ? new Uint8Array(input) : input;
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
            let js = runtime.as_mut().expect("runtime is present while the loop runs");
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
                    if let Some(id) = pending_reply.take() {
                        if sys.call_is_pending(&id) {
                            sys.resolve_call(&id, Err(format!("the handler raised: {e}")));
                        }
                    }
                }
                continue;
            }
            Wake::Ctl(Some(Control::Replace(source))) => {
                ui::trace(&me.tag, "⟳ replacing script code");
                drop(runtime.take());
                match boot(&sys, &me, &instructions, &source).await {
                    Some(fresh) => runtime = Some(fresh),
                    None => return,
                }
                settled = false;
                continue;
            }
            // The sender is dropped when this process is stopped, which is how
            // a blocked script learns to exit.
            Wake::Ctl(None) | Wake::Mail(None) => return,
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
        let js = runtime.as_mut().expect("runtime is present while the loop runs");
        // Only the synchronous part runs here. Whatever the handler leaves
        // pending is driven by the race at the top of the loop, which is what
        // lets a handler await a socket without deafening the process.
        if let Err(e) = js.execute_script("[bitty:mail]", call) {
            let e = e.to_string();
            ui::warn(&me.tag, &format!("handler failed: {e}"));
            // A caller blocked on this message must not wait for the timeout
            // when the handler has already blown up.
            if let Some(id) = pending_reply.take() {
                if sys.call_is_pending(&id) {
                    sys.resolve_call(&id, Err(format!("the handler raised: {e}")));
                }
            }
        }
        settled = false;
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
            // Strip the types before V8 sees it. The precheck transpiles too,
            // but only to validate — running the original source means every
            // type annotation reaches V8 as a syntax error, which reads as
            // "Missing initializer in const declaration" and sends the author
            // hunting for a bug that is not in their code.
            let js = match transpile("inline", &wrapped) {
                Ok(js) => js,
                Err(e) => {
                    sys_t.resolve_call(&id, Err(format!("script error: {e}")));
                    return;
                }
            };
            if let Err(e) = runtime.execute_script("[inline]", js) {
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
        op_bitty_serve(),
        op_bitty_sleep(),
        op_bitty_ws_connect(),
        op_bitty_ws_send(),
        op_bitty_ws_recv(),
        op_bitty_ws_close(),
        op_bitty_env(),
        op_bitty_env_list(),
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
interface BittySocket {
  id: number;
  send(text: string | object): void;
  recv(): Promise<string | null>;
  close(): void;
  [Symbol.asyncIterator](): AsyncIterableIterator<string>;
}
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
  fetch(url: string, opts?: { method?: string; body?: string | object; headers?: Record<string, string> }): { status: number; body: string; headers: Record<string, string> };
  env(name: string): string;
  envNames(): string[];
  sleep(ms: number): Promise<void>;
  connect(url: string): Promise<BittySocket>;
  sys(key: string): string;
}
declare const bitty: BittyApi;

declare class Headers {
  constructor(init?: Record<string, string> | Iterable<[string, string]>);
  get(name: string): string | null;
  set(name: string, value: string): void;
  has(name: string): boolean;
  delete(name: string): void;
  entries(): IterableIterator<[string, string]>;
  forEach(fn: (value: string, name: string, parent: Headers) => void): void;
  toJSON(): Record<string, string>;
  [Symbol.iterator](): IterableIterator<[string, string]>;
}
declare class Request {
  constructor(url: string, init?: { method?: string; headers?: Record<string, string> | Headers; body?: string });
  readonly url: string;
  readonly method: string;
  readonly headers: Headers;
  text(): Promise<string>;
  json(): Promise<any>;
}
declare class Response {
  constructor(body?: string | null, init?: { status?: number; headers?: Record<string, string> | Headers });
  static json(value: unknown, init?: { status?: number; headers?: Record<string, string> | Headers }): Response;
  readonly status: number;
  readonly ok: boolean;
  readonly headers: Headers;
  text(): Promise<string>;
  json(): Promise<any>;
}
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
        // The return type is inferred rather than declared: annotating it
        // Promise<unknown> makes TypeScript demand a return statement, which
        // rejects every inline script that only has side effects.
        format!("async function __inline() {{\n{source}\n}}\nvoid __inline;")
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
