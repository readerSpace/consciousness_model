const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('electronAPI', {
  selectWorkspace: () => ipcRenderer.invoke('select-workspace'),
})
