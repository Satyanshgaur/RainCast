'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Navbar() {
  const pathname = usePathname();

  const links = [
    { name: 'Products', href: '/products' },
    { name: 'Documentation', href: '/docs' },
    { name: 'About', href: '/about' },
    { name: 'Pricing', href: '/pricing' },
  ];

  return (
    <header className="navbar-container">
      <nav className="navbar">
        <Link href="/" className="nav-brand">
          <svg className="brand-logo" width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <polygon points="12,2 2,22 22,22" fill="#ffffff" />
          </svg>
          <span className="brand-name">Raincast</span>
        </Link>
        
        <ul className="nav-links">
          {links.map((link) => {
            const isActive = pathname === link.href;
            return (
              <li key={link.href}>
                <Link
                  href={link.href}
                  className={`nav-link ${isActive ? 'active' : ''}`}
                >
                  {link.name}
                </Link>
              </li>
            );
          })}
        </ul>

        <div className="nav-actions">
          <Link href="/login" className={`nav-link ${pathname === '/login' ? 'active' : ''}`}>
            Log in
          </Link>
          <Link href="/products" className="btn-secondary">
            Launch
          </Link>
        </div>
      </nav>
    </header>
  );
}
