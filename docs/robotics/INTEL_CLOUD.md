# Intel cloud assessment for the sandbox and final hardware

## September 13 access recheck

The rendered portal still advertises Core Ultra AI PCs, browser/SSH access and a self-service Test Drive category. The actual hardware catalog requires sign-in, so no Series 2/3 SKU, available slot, price or approved reservation has been verified for this account.

The [current request guide](https://cloud.intel.com/docs/how_to_request.html) states **2–3 business days** for approved/rejected/waitlisted status; pre-approved requests can be immediate. That delay may exceed our September 14 finish target. Reservations have fixed expiry and a maximum total duration of 30 days. Linux browser access starts JupyterLab; Windows starts RDP. Browser access does not require uploading an SSH key.

There is an account-guidance discrepancy: the portal says corporate **or personal** accounts, while the [registration guide](https://cloud.intel.com/docs/how_to_register.html) still says corporate email. Do not assume either that a personal account is excluded or that it is entitled to a particular machine. The user has been asked to sign in privately so the actual catalog can be inspected. No account was created, agreement accepted, instance requested, or charge incurred by this recheck.

Prepare both routes: an available qualifying AI PC in this cloud, or the user's Intel laptop with hardware eligibility explicitly documented. Running inference on this AMD/RTX development PC does not establish Intel execution. The written brief's stricter deployment sentence is still present in the PDF read September 13; keep the livestream clarification and hardware uncertainty visible.

## Original September 8 assessment

Checked **September 8, 2026** against the currently rendered [Intel Cloud Services portal](https://cloud.intel.com/).

**This is a promising route to request qualifying hardware.** The current portal explicitly offers an **AI PC** category using **Intel Core Ultra processors**, alongside Arc Pro GPU and Xeon server categories. Its workflow says to choose hardware, request an instance, await approval (typically within three days), and connect through a browser or SSH. It also describes Test Drive access as self-service. Exact SKU, availability, duration, quotas, and any costs for this account were not verified.

The old Tiber name/link appears in older documentation. An [Intel moderator response](https://community.intel.com/t5/Edge-Software-Catalog/How-to-access-intel-Tiber-cloud-developer-access/m-p/1722198?profile.language=en) directs users to `cloud.intel.com`; the current portal is branded Intel Cloud Services.

## Recommendation

Request access soon, because the stated approval time can extend past the September 10 kickoff. In the hardware catalog, look specifically for **Core Ultra Series 2 or Series 3**, and record the actual CPU model and available integrated GPU/NPU. Generic Core Ultra branding is not enough to establish the series.

For the final demonstration, the [Intel online brief](https://drive.google.com/file/d/1xSisqTQUAFQiLOpjLZrCVTCsQi4bMCpO/view) specifies that the simulation and robotics AI inference execute on Core Ultra Series 2/3. A Xeon, Gaudi, or standalone Arc instance could be useful for other development/training, but is not evidence that this client-hardware requirement is met. Confirm that remote use of the qualifying machine is accepted for the final demo.

Before committing the architecture, verify:

1. Exact CPU series/model and usable GPU/NPU devices.
2. Access through the submission period and any session/reset limits.
3. Ability to install our Python dependencies and run MuJoCo camera rendering.
4. Model memory and inference performance on that actual machine.
5. How the browser interface can be accessed privately (for example through an approved SSH tunnel).

The local sandbox can proceed while this is arranged. We have not created a cloud account, submitted an access request, provisioned an instance, or incurred cloud charges.
