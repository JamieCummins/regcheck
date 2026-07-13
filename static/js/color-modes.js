/*!
 * RegCheck theme toggler.
 * Dark is the default (RegCheck is dark-first); light is opt-in and persisted.
 * Runs in <head> so the chosen theme is applied before first paint.
 */
(() => {
  'use strict'

  const STORAGE_KEY = 'theme'
  const getStoredTheme = () => {
    try { return localStorage.getItem(STORAGE_KEY) } catch (_e) { return null }
  }
  const setStoredTheme = (theme) => {
    try { localStorage.setItem(STORAGE_KEY, theme) } catch (_e) { /* ignore */ }
  }

  // Default to dark when nothing is stored (we intentionally do NOT follow the
  // OS prefers-color-scheme — dark is RegCheck's default).
  const getPreferredTheme = () => (getStoredTheme() === 'light' ? 'light' : 'dark')

  const applyTheme = (theme) => {
    document.documentElement.setAttribute('data-bs-theme', theme === 'light' ? 'light' : 'dark')
  }

  // Apply immediately (this script is in <head>, before the body paints).
  applyTheme(getPreferredTheme())

  const syncToggles = () => {
    const isLight = getPreferredTheme() === 'light'
    document.querySelectorAll('[data-theme-toggle]').forEach((btn) => {
      btn.setAttribute('aria-pressed', String(isLight))
      btn.setAttribute('aria-label', isLight ? 'Switch to dark theme' : 'Switch to light theme')
      btn.setAttribute('title', isLight ? 'Switch to dark theme' : 'Switch to light theme')
    })
  }

  window.addEventListener('DOMContentLoaded', () => {
    syncToggles()
    document.querySelectorAll('[data-theme-toggle]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const next = getPreferredTheme() === 'dark' ? 'light' : 'dark'
        setStoredTheme(next)
        applyTheme(next)
        syncToggles()
      })
    })
  })
})()
