const { app, BrowserWindow, dialog, ipcMain } = require('electron')
const path = require('path')
const { spawn } = require('child_process')

let bridge

// Some Windows environments cannot initialize Electron's GPU process. The
// coding UI is text-focused, so software rendering is a reliable default.
app.disableHardwareAcceleration()
app.commandLine.appendSwitch('disable-gpu')
app.commandLine.appendSwitch('disable-gpu-compositing')
app.commandLine.appendSwitch('in-process-gpu')
app.commandLine.appendSwitch('disable-gpu-sandbox')

function startBridge() {
  const projectRoot = path.resolve(__dirname, '..', '..', '..')
  bridge = spawn('python', ['-m', 'coding_world_benchmark.coding_agent_bridge', process.env.CODING_AGENT_BRIDGE_PORT || '8787'], {
    cwd: projectRoot, windowsHide: true, stdio: 'ignore',
  })
}

async function registerWorkspace(selectedPath) {
  const port = process.env.CODING_AGENT_BRIDGE_PORT || '8787'
  for (let attempt = 0; attempt < 20; attempt += 1) {
    try {
      const response = await fetch(`http://127.0.0.1:${port}/api/workspace`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: selectedPath }),
      })
      const data = await response.json()
      if (!response.ok || !data.workspace) throw new Error(data.error || `bridge error (${response.status})`)
      return data.workspace
    } catch (error) {
      if (attempt === 19) throw error
      await new Promise((resolve) => setTimeout(resolve, 200))
    }
  }
}

function createWindow() {
  const window = new BrowserWindow({
    width: 1440, height: 900, minWidth: 920, minHeight: 620,
    backgroundColor: '#eef1f4',
    show: false,
    webPreferences: { preload: path.join(__dirname, 'preload.cjs'), contextIsolation: true, nodeIntegration: false },
  })
  window.once('ready-to-show', () => window.show())
  window.webContents.on('did-fail-load', (_event, code, description) => {
    dialog.showErrorBox('UIを読み込めません', `${description} (${code})`)
  })
  window.loadFile(path.join(__dirname, '..', 'dist', 'index.html'))
}

ipcMain.handle('select-workspace', async () => {
  const result = await dialog.showOpenDialog({ title: 'コーディング作業フォルダを選択', properties: ['openDirectory'] })
  if (result.canceled) return null
  return registerWorkspace(result.filePaths[0])
})

app.whenReady().then(() => {
  startBridge()
  createWindow()
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow() })
})
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit() })
app.on('will-quit', () => { if (bridge && !bridge.killed) bridge.kill() })
