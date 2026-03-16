# SPIRAL Remote Bootstrap

These helper scripts prepare and launch the current Phase XVI boundary sweep on a remote GPU box.

Files:

- `bootstrap_spiral_b300.sh`
- `sync_spiral_to_b300.sh`
- `run_b300_boundary_sweep.sh`

Typical flow:

```bash
ssh -i ~/.ssh/id_ed25519_codex_external_20260314 ubuntu@HOST
bash /path/to/bootstrap_spiral_b300.sh ~/spiral_runtime
```

Then locally:

```bash
bash /mnt/d/spiral_remote/sync_spiral_to_b300.sh ubuntu@HOST ~/spiral_runtime
bash /mnt/d/spiral_remote/run_b300_boundary_sweep.sh ubuntu@HOST ~/spiral_runtime 1000,2000,5000
```
