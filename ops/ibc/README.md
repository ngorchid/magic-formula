# IBC wrapper copies

Reference copies of the hand-written wrappers that live in the IBC folders on the trading PC
(`C:\Users\Nicolas\IBC\...`). Those folders are not in git or the OneDrive backup; these copies are
how to rebuild them. **The PC's copy is the one that runs**: after editing it, copy it back here
and commit.

| File | Runs from | Used by |
|---|---|---|
| `IBCWin-live/StartGatewayFresh.bat` | `C:\Users\Nicolas\IBC\IBCWin-live\` | `scripts/check_live_gateway.py` (auto-recovery of the live Gateway, port 4001) |

IBC config files (`config.ini`) are deliberately NOT copied: they hold the IB login.
