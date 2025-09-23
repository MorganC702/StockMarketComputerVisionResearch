// import { Link } from 'react-router-dom'
import type { JSX } from 'react'

export default function Footer(): JSX.Element {
    return (
        <footer className="bg-zinc-950 border-t border-cyan-900 py-4">
            <div className="max-w-7xl mx-auto px-6 lg:px-8">
                <div className="text-center text-xs text-gray-500">
                    &copy; {new Date().getFullYear()} Morgan Cooper. All rights reserved.
                </div>
            </div>
        </footer>
    )
}
