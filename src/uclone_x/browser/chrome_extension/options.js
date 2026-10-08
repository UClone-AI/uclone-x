// The pairing page: one field for the code Settings ▸ Browser shows, and the link's state in words.

const CODE = /^\s*\d{2,5}-[0-9a-f]{16,128}\s*$/i;
const statusLine = document.getElementById("status");
const field = document.getElementById("code");

for (const node of document.querySelectorAll("[data-i18n]")) {
  node.textContent = chrome.i18n.getMessage(node.dataset.i18n);
}

function show(status) {
  statusLine.dataset.status = status;
  statusLine.textContent = chrome.i18n.getMessage(`status_${status}`) || chrome.i18n.getMessage("status_unpaired");
}

chrome.storage.local.get(["pairingCode", "status"]).then(({ pairingCode, status }) => {
  if (pairingCode) field.value = pairingCode;
  show(status || "unpaired");
});

chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && changes.status) show(changes.status.newValue || "unpaired");
});

document.getElementById("pair").addEventListener("submit", async (event) => {
  event.preventDefault();
  const code = field.value.trim();
  if (!CODE.test(code)) {
    statusLine.dataset.status = "not_a_code";
    statusLine.textContent = chrome.i18n.getMessage("notACode");
    return;
  }
  await chrome.storage.local.set({ pairingCode: code });
  chrome.runtime.sendMessage({ type: "redial" });
});
