import { Component } from 'react'

// Catches render errors (e.g. WebGL unavailable) so the lazy 3D view can fall
// back to a message instead of taking down the page.
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { failed: false }
  }
  static getDerivedStateFromError() {
    return { failed: true }
  }
  render() {
    if (this.state.failed) {
      return (
        this.props.fallback ?? (
          <div className="rounded-lg border border-zinc-300 bg-zinc-50 p-4 text-sm text-zinc-500 dark:border-zinc-700 dark:bg-zinc-900">
            3D view unavailable (WebGL may be disabled). The 2D plots above still prove the claim.
          </div>
        )
      )
    }
    return this.props.children
  }
}
