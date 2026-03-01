document.addEventListener('DOMContentLoaded', () => {
  const { animate, stagger } = window.motion || {}

  // Container entrance
  const container = document.querySelector('.motion-container')
  if (container && animate) {
    animate(container, { opacity: [0, 1], transform: ['translateY(18px)', 'translateY(0px)'] }, { duration: 0.55, easing: 'ease' })
    // stagger children
    const children = Array.from(container.querySelectorAll('.glass-card, .dropzone'))
    children.forEach((el, i) => {
      animate(el, { opacity: [0, 1], transform: ['translateY(8px)', 'translateY(0px)'] }, { delay: 0.08 * i, duration: 0.45 })
    })
  }

  // Dropzone interactions
  const dropZone = document.getElementById('dropZone')
  const pdfInput = document.getElementById('pdfInput')
  const browseLink = document.getElementById('browseLink')
  const fileInfo = document.getElementById('fileInfo')
  const fileNameEl = document.getElementById('fileName')
  const removeFileBtn = document.getElementById('removeFile')
  const processBtn = document.getElementById('processBtn')
  const loadingEl = document.getElementById('loading')
  const resultsCard = document.getElementById('resultsCard')
  const mdOutput = document.getElementById('mdOutput')
  const errorBox = document.getElementById('errorBox')

  function setFile(file) {
    if (!file) return
    fileInfo.classList.remove('hidden')
    fileNameEl.textContent = file.name
    processBtn.disabled = false
  }

  function clearFile() {
    fileInfo.classList.add('hidden')
    fileNameEl.textContent = ''
    pdfInput.value = ''
    processBtn.disabled = true
  }

  dropZone.addEventListener('click', () => pdfInput.click())
  browseLink.addEventListener('click', (e) => { e.stopPropagation(); pdfInput.click() })

  ;['dragenter','dragover'].forEach(evt => {
    dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); if (animate) animate(dropZone, { transform: ['translateY(0px)','translateY(-3px)'] }, { duration: 0.18 }) })
  })
  ;['dragleave','dragend','drop'].forEach(evt => {
    dropZone.addEventListener(evt, (e) => { dropZone.classList.remove('drag-over'); if (evt === 'drop') {
      e.preventDefault(); const f = (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) || null; if (f) { pdfInput.files = e.dataTransfer.files; setFile(f) }
    } })
  })

  pdfInput.addEventListener('change', (e) => {
    const f = e.target.files && e.target.files[0]
    if (f) setFile(f)
  })

  removeFileBtn && removeFileBtn.addEventListener('click', (e) => { e.preventDefault(); clearFile() })

  // Process button: show loading and then reveal results area only (do not break existing handlers)
  processBtn && processBtn.addEventListener('click', async (e) => {
    // If page has other handlers, let them run; still show polished UI
    errorBox.classList.add('hidden')
    loadingEl.classList.remove('hidden')
    processBtn.disabled = true
    // subtle simulated shimmer before real backend responds; remove after 12s or when results injected by backend
    const fallback = setTimeout(() => {
      loadingEl.classList.add('hidden')
      resultsCard.classList.remove('hidden')
      // simple placeholder if backend hasn't injected md
      if (mdOutput && mdOutput.innerHTML.trim() === '') mdOutput.innerHTML = '<p class="text-slate-600">Study plan will appear here once processing completes.</p>'
    }, 12000)

    // Allow other page scripts to run; listen for a custom event `studyPlanReady` to cancel fallback
    const onReady = () => {
      clearTimeout(fallback)
      loadingEl.classList.add('hidden')
      resultsCard.classList.remove('hidden')
      processBtn.disabled = false
      document.removeEventListener('studyPlanReady', onReady)
    }
    document.addEventListener('studyPlanReady', onReady)
  })

  // Allow external code to trigger a polished reveal by dispatching `studyPlanReady`

  // Basic accessibility: keyboard on dropzone
  dropZone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pdfInput.click() } })

  // Optional animated count-up helper (used when backend injects numbers)
  function countUp(el, to) {
    if (!el) return
    const start = 0
    const duration = 700
    const startTime = performance.now()
    function step(now){
      const t = Math.min(1, (now - startTime)/duration)
      el.textContent = Math.round(start + (to - start) * t)
      if (t < 1) requestAnimationFrame(step)
    }
    requestAnimationFrame(step)
  }

  // Listen for backend to inject MD and counts. The backend may set innerHTML on #mdOutput and then dispatch 'studyPlanReady' event.
  // Provide a convenience: if mdOutput receives children, animate them in.
  const observer = new MutationObserver((mutations) => {
    for (const m of mutations) {
      if (m.type === 'childList' && m.addedNodes.length) {
        // animate newly added nodes
        if (animate) {
          const nodes = Array.from(m.addedNodes).filter(n => n.nodeType === 1)
          nodes.forEach((n,i) => animate(n, { opacity: [0,1], transform: ['translateY(6px)','translateY(0px)'] }, { delay: i*0.06, duration: 0.32 }))
        }
        // dispatch ready so processBtn handler clears fallback
        document.dispatchEvent(new Event('studyPlanReady'))
      }
    }
  })
  if (mdOutput) observer.observe(mdOutput, { childList: true, subtree: true })

})
