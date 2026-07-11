// Injected by ComfyUI auto-download middleware.
// Routes model downloads through Python backend and shows visual progress.
;(function () {
  if (window.__comfyDesktop2_dl) return
  window.__comfyDesktop2_dl = true

  let active_count = 0
  const banner = document.createElement('div')
  banner.id = '__comfy_dl_banner'
  Object.assign(banner.style, {
    position: 'fixed', top: '0', left: '0', right: '0', zIndex: '99999',
    background: '#1a1a2e', color: '#e0e0e0', padding: '10px 20px',
    fontFamily: 'system-ui, sans-serif', fontSize: '13px',
    display: 'none', borderBottom: '2px solid #4a9eff',
    boxShadow: '0 2px 12px rgba(0,0,0,0.5)',
  })
  ;(function wait() {
    if (document.body && !document.getElementById('__comfy_dl_banner')) document.body.prepend(banner)
    if (!document.getElementById('__comfy_dl_banner')) setTimeout(wait, 200)
  })()

  function fmt_bytes(b) {
    if (!b || b <= 0) return ''
    if (b >= 1e9) return (b / 1e9).toFixed(1) + ' GB'
    if (b >= 1e6) return (b / 1e6).toFixed(1) + ' MB'
    if (b >= 1e3) return (b / 1e3).toFixed(0) + ' KB'
    return b + ' B'
  }

  window.__comfyDesktop2 = {
    isRemote: () => false,
    downloadModel: async function (url, name, directory) {
      const label = directory + '/' + name
      active_count++
      banner.innerHTML = `<span style="font-weight:600">&#x2B07; ${label} &mdash; starting...</span>`
      banner.style.display = 'flex'
      try {
        const resp = await fetch('/api/download-model', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ folder_type: directory, filename: name, url: url }),
        })
        const data = await resp.json()
        if (data.status === 'downloaded') { active_count--; if (active_count <= 0) banner.style.display = 'none'; window.location.reload(); return }
        if (data.status === 'started' || data.status === 'downloading') await poll_download(data.key, label)
        else { active_count--; if (active_count <= 0) banner.style.display = 'none' }
      } catch (e) { active_count--; if (active_count <= 0) banner.style.display = 'none' }
    },
  }

  async function poll_download(key, label) {
    let start = Date.now(), last_bytes = 0
    for (let i = 0; i < 600; i++) {
      await new Promise(r => setTimeout(r, 1000))
      try {
        const resp = await fetch('/api/download-status/' + encodeURIComponent(key))
        const data = await resp.json()
        if (data.status === 'downloaded') { active_count--; if (active_count <= 0) banner.style.display = 'none'; window.location.reload(); return }
        if (data.status === 'failed') {
          banner.innerHTML = `<span style="color:#ff6b6b;font-weight:600">&#x2716; ${label}: ${data.reason || 'failed'}</span><button onclick="document.getElementById('__comfy_dl_banner').style.display='none'" style="margin-left:auto;background:none;border:1px solid #555;color:#ccc;padding:2px 10px;cursor:pointer;border-radius:3px">Dismiss</button>`
          return
        }

        let dl = data.downloaded_bytes || 0, total = data.total_bytes || 0
        let elapsed = (Date.now() - start) / 1000
        let speed = elapsed > 0 ? dl / elapsed : 0
        let remain = speed > 0 && total > 0 ? (total - dl) / speed : 0
        let pct = total > 0 ? Math.round(dl * 100 / total) : 0

        let info = '<span style="font-weight:600">&#x2B07; ' + label + '</span>'
        if (total > 0) {
          info += '<span style="margin-left:12px;color:#aaa">' + fmt_bytes(dl) + ' / ' + fmt_bytes(total) + ' (' + pct + '%)</span>'
          info += '<span style="margin-left:12px;color:#aaa">' + fmt_bytes(speed) + '/s</span>'
          info += '<span style="margin-left:12px;color:#aaa">~' + fmt_duration(remain) + ' left</span>'
          let bar = document.createElement('div')
          bar.style.cssText = 'flex:1;height:4px;background:#333;border-radius:2px;margin-left:12px;overflow:hidden'
          bar.innerHTML = '<div style="height:100%;width:' + pct + '%;background:#4a9eff;border-radius:2px;transition:width 0.5s"></div>'
          banner.innerHTML = ''
          banner.appendChild(document.createTextNode(''))
          banner.innerHTML = info
          banner.appendChild(bar)
        } else {
          info += '<span style="margin-left:12px;color:#aaa">' + fmt_bytes(dl) + ' downloaded</span>'
          info += '<span style="margin-left:12px;color:#aaa">' + fmt_bytes(speed) + '/s</span>'
          banner.innerHTML = info
        }
      } catch (e) {}
    }
  }

  function fmt_duration(s) {
    if (!s || s < 0 || !isFinite(s)) return '...'
    if (s < 60) return Math.round(s) + 's'
    if (s < 3600) return Math.round(s / 60) + 'm'
    return Math.round(s / 3600) + 'h ' + Math.round((s % 3600) / 60) + 'm'
  }
})()
