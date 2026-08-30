# run_exec_iso_overnight.ps1
# Execution-model isolation for dissertation issue #2 (the target-selection confound).
# Runs flat DF best-of-K on maze2d-large at MATCHED cadence (--replan-every 1),
# varying ONLY the target rule:
#     --reach-wp 0.0  -> aim at the immediate next waypoint  ("aim-next")
#     --reach-wp 1.0  -> advance past reached waypoints        ("advance-past")
# 3 seeds each. This splits the ~55-point per-step-vs-rp50 gap into cadence vs rule.
#
# Safety: quick preflight, a 7.25h global deadline, and a 90-min per-run kill.
# Total wall-clock is bounded under 8 hours. Safe to leave overnight.

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
Set-Location "D:\Surrey\Sem 2\Dissertation\backup\mcts_diffusion_back"

$logDir = "results\exec_iso_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$perRunCapSec = 5400          # 90 min hard cap per run
$budgetHours  = 7.25          # real-run budget (preflight adds <=15 min on top)

function Start-Py([string[]]$pyArgs, [string]$logBase, [int]$timeoutSec) {
    $p = Start-Process -FilePath "python" -ArgumentList $pyArgs -NoNewWindow -PassThru `
         -RedirectStandardOutput "$logBase.log" -RedirectStandardError "$logBase.err"
    if (-not $p.WaitForExit($timeoutSec * 1000)) {
        try { $p.Kill() } catch {}
        return $null   # timed out
    }
    return $p.ExitCode
}

# ---- preflight: validate the whole pipeline quickly (<=15 min) ---------------
Write-Host "[PREFLIGHT] 2-env / 50-step validation..."
$pf = @("scripts/run_mctd.py","--env","maze2d-large-v1","--flat-mcss","--mcss-backbone","df",
        "--k","50","--replan-every","1","--reach-wp","1.0","--seed","0",
        "--n-envs","2","--max-steps","50","--out","results/exec_iso_preflight.json")
$pfExit = Start-Py $pf "$logDir\preflight" 900
if ($null -eq $pfExit -or -not (Test-Path "results/exec_iso_preflight.json")) {
    Write-Host "[PREFLIGHT-FAIL] no valid output (see $logDir\preflight.err). Aborting so the night is not wasted."
    exit 1
}
Write-Host "[PREFLIGHT-OK] pipeline works. Starting the real runs."

# ---- real runs, ordered so each seed's PAIR finishes before the next seed ----
$deadline = (Get-Date).AddHours($budgetHours)
$jobs = @(
  @{ rw = "1.0"; s = 0; out = "results\exec_iso_advance_s0.json"; base = "$logDir\advance_s0"; tag = "advance s0" },
  @{ rw = "0.0"; s = 0; out = "results\exec_iso_aimnext_s0.json"; base = "$logDir\aimnext_s0"; tag = "aimnext s0" },
  @{ rw = "1.0"; s = 1; out = "results\exec_iso_advance_s1.json"; base = "$logDir\advance_s1"; tag = "advance s1" },
  @{ rw = "0.0"; s = 1; out = "results\exec_iso_aimnext_s1.json"; base = "$logDir\aimnext_s1"; tag = "aimnext s1" },
  @{ rw = "1.0"; s = 2; out = "results\exec_iso_advance_s2.json"; base = "$logDir\advance_s2"; tag = "advance s2" },
  @{ rw = "0.0"; s = 2; out = "results\exec_iso_aimnext_s2.json"; base = "$logDir\aimnext_s2"; tag = "aimnext s2" }
)

foreach ($j in $jobs) {
    $remaining = [int]($deadline - (Get-Date)).TotalSeconds
    if ($remaining -le 180) { Write-Host "[BUDGET] Out of time - skipping $($j.tag) and the rest."; break }
    $timeoutSec = [Math]::Min($perRunCapSec, $remaining)
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] START $($j.tag)  (timeout ${timeoutSec}s, budget ends $($deadline.ToString('HH:mm')))"
    $pyArgs = @("scripts/run_mctd.py","--env","maze2d-large-v1","--flat-mcss","--mcss-backbone","df",
                "--k","50","--replan-every","1","--reach-wp",$j.rw,"--seed","$($j.s)","--out",$j.out)
    $exit = Start-Py $pyArgs $j.base $timeoutSec
    if ($null -eq $exit) {
        Write-Host "[TIMEOUT] $($j.tag) exceeded ${timeoutSec}s - killed, moving on."
    } else {
        Write-Host "[DONE] $($j.tag)  exit=$exit  outputWritten=$(Test-Path $j.out)"
    }
}

Write-Host "[FINISHED] $(Get-Date -Format 'HH:mm:ss'). Results:"
Get-ChildItem "results\exec_iso_*.json" -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $($_.Name)" }
