import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { Bell } from 'lucide-react';
import Sidebar from './Sidebar';
import { getUnreadCount } from '../api/client';

// Lazy-import pages that run long tasks so they stay mounted across navigation
import StrategyConfig from '../pages/StrategyConfig';
import FactorMining from '../pages/factor-mining';
import AutoTrade from '../pages/AutoTrade';

// Pages that should stay mounted to preserve running-task state
const PERSISTENT_PAGES: { path: string; element: React.ReactNode }[] = [
  { path: '/strategy', element: <StrategyConfig /> },
  { path: '/factor-mining', element: <FactorMining /> },
  { path: '/autotrade', element: <AutoTrade /> },
];

export default function Layout() {
  const [unread, setUnread] = useState(0);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    const poll = () => getUnreadCount().then(d => setUnread(d?.unread_count ?? 0)).catch(() => {});
    poll();
    const id = setInterval(poll, 15000);
    return () => clearInterval(id);
  }, []);

  const currentPath = location.pathname;
  const isPersistentPage = PERSISTENT_PAGES.some(p => p.path === currentPath);

  return (
    <div className="min-h-screen bg-[#0f172a]">
      <Sidebar />
      <div className="pl-60 h-screen flex flex-col">
        {/* Top bar with notification bell */}
        <div className="flex items-center justify-end px-6 py-2 border-b border-[#1e293b]">
          <button onClick={() => navigate('/notifications')}
            className="relative p-2 rounded-lg hover:bg-[#1e293b] text-[#94a3b8] hover:text-[#f8fafc] transition-colors">
            <Bell className="h-5 w-5" />
            {unread > 0 && (
              <span className="absolute -top-0.5 -right-0.5 min-w-[18px] h-[18px] flex items-center justify-center rounded-full bg-red-500 text-white text-[10px] font-bold px-1">
                {unread > 99 ? '99+' : unread}
              </span>
            )}
          </button>
        </div>
        <main className="flex-1 overflow-y-auto p-6">
          <div className="mx-auto max-w-7xl">
            {/* Persistent pages: always mounted, hidden when not active */}
            {PERSISTENT_PAGES.map(p => (
              <div key={p.path} className={currentPath === p.path ? '' : 'hidden'}>
                {p.element}
              </div>
            ))}
            {/* Other pages: normal Outlet rendering */}
            {!isPersistentPage && <Outlet />}
          </div>
        </main>
      </div>
    </div>
  );
}
