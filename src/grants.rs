//! Capability grants: what a process is allowed to do, and to whom.
//!
//! One representation covers every permissioned verb, so adding a capability
//! is a new enum variant rather than another bespoke field with its own
//! interpretation of "unrestricted". A grant is always one of: everything,
//! an explicit set of process ids, or nothing.
//!
//! The load-bearing rule is **attenuation**: a spawned process can never hold
//! a grant its spawner lacks. Without that, any restricted process could spawn
//! an unrestricted child and use it as a proxy, so isolation would only ever
//! be one spawn deep.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Capability {
    Send,
    Stop,
    Spawn,
    /// Programs a process may execute. Targets are command names, not process
    /// ids — the same `Grant` shape, a different namespace.
    Run,
    /// Hosts a process may reach over the network.
    Net,
    /// Environment variables it may read.
    Env,
    /// System facts it may query (hostname, osRelease, uid).
    Sys,
}

impl Capability {
    pub const ALL: [Capability; 7] = [
        Capability::Send,
        Capability::Stop,
        Capability::Spawn,
        Capability::Run,
        Capability::Net,
        Capability::Env,
        Capability::Sys,
    ];

    /// How this capability reads in a permissions summary.
    pub fn verb(self) -> &'static str {
        match self {
            Capability::Send => "Messaging",
            Capability::Stop => "Stopping",
            Capability::Spawn => "Spawning",
            Capability::Run => "Running",
            Capability::Net => "Network",
            Capability::Env => "Environment",
            Capability::Sys => "System info",
        }
    }

    /// Whether naming specific targets is meaningful. Spawning creates a new
    /// process, so there is nothing to name — it is allowed or it isn't.
    pub fn targets_processes(self) -> bool {
        matches!(self, Capability::Send | Capability::Stop)
    }

    /// How an unrestricted grant of this capability reads.
    fn everything(self) -> &'static str {
        match self {
            Capability::Send | Capability::Stop => "any process",
            Capability::Spawn => "permitted",
            Capability::Run => "any program",
            Capability::Net => "any host",
            Capability::Env => "any variable",
            Capability::Sys => "permitted",
        }
    }

    fn nothing(self) -> &'static str {
        match self {
            Capability::Send | Capability::Stop => "no one",
            _ => "not permitted",
        }
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Grant {
    /// Any process, including ones that don't exist yet.
    All,
    /// Exactly these ids (already resolved; may include the literal "user").
    Ids(HashSet<String>),
    /// Nothing at all.
    Nobody,
}

impl Grant {
    pub fn permits(&self, target: &str) -> bool {
        match self {
            Grant::All => true,
            Grant::Ids(ids) => ids.contains(target),
            Grant::Nobody => false,
        }
    }

    /// True when this grant allows anything at all.
    pub fn is_permissive(&self) -> bool {
        !matches!(self, Grant::Nobody)
    }

    pub fn ids(&self) -> Option<&HashSet<String>> {
        match self {
            Grant::Ids(ids) => Some(ids),
            _ => None,
        }
    }

    /// Clamp this grant to what the spawner itself holds. Pure narrowing —
    /// the result never permits anything the ceiling doesn't.
    pub fn attenuate(self, ceiling: &Grant) -> Grant {
        match ceiling {
            Grant::All => self,
            Grant::Nobody => Grant::Nobody,
            Grant::Ids(ceiling_ids) => match self {
                Grant::Nobody => Grant::Nobody,
                // "everything" means "everything the spawner could reach".
                Grant::All => Grant::Ids(ceiling_ids.clone()),
                Grant::Ids(mine) => {
                    let kept: HashSet<String> = mine.intersection(ceiling_ids).cloned().collect();
                    if kept.is_empty() {
                        Grant::Nobody
                    } else {
                        Grant::Ids(kept)
                    }
                }
            },
        }
    }

    /// Add ids that hold regardless of the ceiling. These are the structural
    /// invariants, not privileges: a process may always answer whoever spawned
    /// it, and may always stop itself. Neither widens anyone's authority, and
    /// both must survive attenuation — a process that cannot stop itself can
    /// never exit cleanly.
    pub fn with(self, extra: &[String]) -> Grant {
        if extra.is_empty() {
            return self;
        }
        match self {
            Grant::All => Grant::All,
            Grant::Nobody => Grant::Ids(extra.iter().cloned().collect()),
            Grant::Ids(mut ids) => {
                ids.extend(extra.iter().cloned());
                Grant::Ids(ids)
            }
        }
    }

    /// Ids that were requested but denied by attenuation, for an honest
    /// explanation at spawn time.
    pub fn dropped_from(&self, requested: &Grant) -> Vec<String> {
        let Some(wanted) = requested.ids() else {
            return Vec::new();
        };
        let mut dropped: Vec<String> = wanted
            .iter()
            .filter(|id| !self.permits(id))
            .cloned()
            .collect();
        dropped.sort();
        dropped
    }
}

/// Filesystem authority, scoped to directory roots rather than to process
/// ids. Modeled on Deno's `--allow-read` / `--allow-write` so the two can be
/// mapped onto each other, but enforced by this harness so it composes with
/// the rest of the capability model — notably attenuation across spawn.
///
/// Roots are canonicalized when granted, and every access is canonicalized
/// before it is checked, so `..`, symlinks, and relative paths cannot escape.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PathGrant {
    Nowhere,
    Under(Vec<PathBuf>),
}

impl PathGrant {
    pub fn is_permissive(&self) -> bool {
        matches!(self, PathGrant::Under(roots) if !roots.is_empty())
    }

    /// The canonical path to use, or why it is refused. `creating` allows a
    /// path that does not exist yet by checking its parent instead — a write
    /// target has to be admissible before it exists.
    pub fn resolve(&self, path: &str, creating: bool) -> Result<PathBuf, String> {
        let PathGrant::Under(roots) = self else {
            return Err(format!("no filesystem access is granted (refused '{path}')"));
        };
        let raw = Path::new(path);
        let canonical = if creating {
            let parent = raw.parent().unwrap_or(Path::new("."));
            let base = parent
                .canonicalize()
                .map_err(|e| format!("cannot resolve '{}': {e}", parent.display()))?;
            match raw.file_name() {
                Some(name) => base.join(name),
                None => return Err(format!("'{path}' does not name a file")),
            }
        } else {
            raw.canonicalize()
                .map_err(|e| format!("cannot resolve '{path}': {e}"))?
        };

        if roots.iter().any(|root| canonical.starts_with(root)) {
            Ok(canonical)
        } else {
            // Deliberately does not say what *is* reachable beyond the roots
            // already named in this process's own prompt.
            Err(format!(
                "'{path}' is outside the directories you may use ({})",
                describe_roots(roots)
            ))
        }
    }

    /// Narrow to what the granter holds: every requested root must sit inside
    /// one the granter already has.
    pub fn attenuate(self, ceiling: &PathGrant) -> PathGrant {
        let PathGrant::Under(bounds) = ceiling else {
            return PathGrant::Nowhere;
        };
        match self {
            PathGrant::Nowhere => PathGrant::Nowhere,
            PathGrant::Under(mine) => {
                let kept: Vec<PathBuf> = mine
                    .into_iter()
                    .filter(|root| bounds.iter().any(|bound| root.starts_with(bound)))
                    .collect();
                if kept.is_empty() {
                    PathGrant::Nowhere
                } else {
                    PathGrant::Under(kept)
                }
            }
        }
    }

    /// Roots that were asked for but fall outside the ceiling.
    pub fn dropped_from(&self, requested: &PathGrant) -> Vec<String> {
        let PathGrant::Under(wanted) = requested else {
            return Vec::new();
        };
        let kept: &[PathBuf] = match self {
            PathGrant::Under(kept) => kept,
            PathGrant::Nowhere => &[],
        };
        wanted
            .iter()
            .filter(|root| !kept.contains(root))
            .map(|root| root.display().to_string())
            .collect()
    }

    pub fn describe(&self) -> String {
        match self {
            PathGrant::Nowhere => "nothing".into(),
            PathGrant::Under(roots) => describe_roots(roots),
        }
    }
}

fn describe_roots(roots: &[PathBuf]) -> String {
    let mut named: Vec<String> = roots.iter().map(|r| r.display().to_string()).collect();
    named.sort();
    named.join(", ")
}

/// The full permission set carried by a process. Named fields rather than a
/// map so adding a capability is a compile error at every site that must
/// handle it, instead of a silently-missing key.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Grants {
    pub send: Grant,
    pub stop: Grant,
    pub spawn: Grant,
    /// Programs that may be executed, by name. `Nobody` by default: running a
    /// command is the widest authority here, because a program's own reach is
    /// not bounded by anything this harness enforces.
    pub run: Grant,
    pub net: Grant,
    pub env: Grant,
    pub sys: Grant,
    pub read: PathGrant,
    pub write: PathGrant,
}

impl Grants {
    /// The authority of the human at the console — the ceiling the root
    /// process is granted from, never the default for a spawned process.
    /// Filesystem reach is bounded by the `--allow-read` / `--allow-write`
    /// flags, which is what root actually receives; this only says the console
    /// is permitted to hand those out.
    pub fn console_authority() -> Self {
        Grants {
            read: PathGrant::Under(vec![PathBuf::from("/")]),
            write: PathGrant::Under(vec![PathBuf::from("/")]),
            run: Grant::All,
            net: Grant::All,
            env: Grant::All,
            sys: Grant::All,
            ..Grants::unrestricted()
        }
    }

    /// Every in-harness capability, and no filesystem access.
    pub fn unrestricted() -> Self {
        Grants {
            send: Grant::All,
            stop: Grant::All,
            spawn: Grant::All,
            run: Grant::Nobody,
            net: Grant::Nobody,
            env: Grant::Nobody,
            sys: Grant::Nobody,
            // The filesystem is the one authority that is *not* granted by
            // default. Everything else a process can reach is inside this
            // harness; the filesystem is outside it, so it must be asked for.
            read: PathGrant::Nowhere,
            write: PathGrant::Nowhere,
        }
    }

    pub fn get(&self, cap: Capability) -> &Grant {
        match cap {
            Capability::Send => &self.send,
            Capability::Stop => &self.stop,
            Capability::Spawn => &self.spawn,
            Capability::Run => &self.run,
            Capability::Net => &self.net,
            Capability::Env => &self.env,
            Capability::Sys => &self.sys,
        }
    }

    /// True when every capability is unrestricted and no filesystem access is
    /// held — the shape that needs no permissions section in the prompt.
    pub fn is_unrestricted(&self) -> bool {
        Capability::ALL
            .iter()
            .all(|cap| matches!(self.get(*cap), Grant::All))
            && !self.read.is_permissive()
            && !self.write.is_permissive()
    }

    /// Render for the process's own system prompt, using `label` to turn
    /// process ids into something a model can act on. Non-process capabilities
    /// name programs, hosts or variables, which need no lookup.
    pub fn describe(&self, label: &impl Fn(&str) -> String) -> String {
        let mut lines = Vec::new();
        for cap in Capability::ALL {
            let text = match self.get(cap) {
                Grant::All => cap.everything().to_string(),
                Grant::Nobody => cap.nothing().to_string(),
                Grant::Ids(ids) => {
                    let mut named: Vec<String> = ids
                        .iter()
                        .map(|id| if cap.targets_processes() { label(id) } else { id.clone() })
                        .collect();
                    named.sort();
                    format!("only {}", named.join(", "))
                }
            };
            lines.push(format!("- {}: {text}", cap.verb()));
        }
        if self.read.is_permissive() || self.write.is_permissive() {
            lines.push(format!("- Reading files under: {}", self.read.describe()));
            lines.push(format!("- Writing files under: {}", self.write.describe()));
        }
        lines.join("\n")
    }
}
