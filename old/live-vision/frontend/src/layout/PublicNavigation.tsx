'use client'

import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
// eslint-disable-next-line @typescript-eslint/no-unused-vars
import { Bars3Icon, XMarkIcon } from '@heroicons/react/24/outline'

interface AppNavigationProps {
  inner_content: React.ReactNode
}

const PublicNavigation: React.FC<AppNavigationProps> = ({ inner_content }) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  useEffect(() => {
    document.body.style.overflow = mobileMenuOpen ? 'hidden' : ''
    return () => {
      document.body.style.overflow = ''
    }
  }, [mobileMenuOpen])

  return (
    <>
      <header className="bg-zinc-950 border-b border-cyan-900 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto flex items-center px-4 sm:px-6 py-4 justify-between">
          {/* Logo / Label */}
          <Link
            to="/"
            className="text-2xl text-white font-serif font-bold relative inline-block hover:text-cyan-400"
          >
            MRC
          </Link>

          {/* Mobile toggle button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="sm:hidden text-white"
          >
            {mobileMenuOpen ? (
              <XMarkIcon className="size-6" />
            ) : (
              <></>
              // <Bars3Icon className="size-6" />
            )}
          </button>
        </div>

        {/* Mobile menu (empty for now, just keeps structure) */}
        {/* {mobileMenuOpen && (
          <div className="sm:hidden bg-zinc-900 border-t border-cyan-900 px-6 py-6 space-y-6 overflow-y-auto max-h-[calc(100vh-4rem)]">
            <p className="text-cyan-400">Menu placeholder</p>
          </div>
        )} */}
      </header>

      <main className="bg-zinc-950 min-h-screen text-white">
        <div>{inner_content}</div>
      </main>
    </>
  )
}

export default PublicNavigation
