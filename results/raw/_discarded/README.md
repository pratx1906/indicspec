# Discarded runs (kept, never used for results)

| Run | Why discarded | Decision |
|---|---|---|
| `pilot_vllm_4b_silent_55W` | Windows power plan "Silent" capped the GPU at 55 W enforced limit (SM clock median 277 MHz of 3,090; `sw_power_cap` active on every busy sample) during the A0 block; the cap changed to 150 W mid-pilot, so arms would have run under different power states. Speed/energy invalid; acceptance unaffected but not used. | D-018 |
