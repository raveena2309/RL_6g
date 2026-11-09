$runs = @(
  "--algo ppo --timesteps 500000 --seeds 5 --train_load_scale 1.6 --capacity_bound",
  "--algo ppo --timesteps 500000 --seeds 5 --train_load_scale 1.6 --lagrangian --capacity_bound",
  "--algo ppo --timesteps 500000 --seeds 5 --train_load_scale 1.6 --outage_prob 0.3",
  "--algo ppo --timesteps 500000 --seeds 5 --train_load_scale 1.6 --lagrangian --capacity_bound --outage_prob 0.3",
  "--algo dqn --timesteps 300000 --seeds 5 --train_load_scale 1.6 --reward_weights 1.0 0.1 0.2 0.3 --lagrangian --budget_lat 0.0005 --budget_loss 0.0005 --lag_lr 50 --suffix _weakw"
)
foreach ($r in $runs) {
  Write-Host "=== $r ==="
  Invoke-Expression "python scripts\train_sb3.py $r" | Tee-Object -FilePath results\queue_log.txt -Append
}