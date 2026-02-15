import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  TrendingUp,
  Settings,
  FlaskConical,
  Wallet,
  MessageSquare,
  Search,
  Bot,
  Newspaper,
  Brain,
  ShieldCheck,
  ScrollText,
} from 'lucide-react';

const navItems = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/market', label: 'Market Data', icon: TrendingUp },
  { to: '/strategy', label: 'Strategy', icon: Settings },
  { to: '/backtest', label: 'Backtest', icon: FlaskConical },
  { to: '/screener', label: '智能选股', icon: Search },
  { to: '/autotrade', label: '自动交易', icon: Bot },
  { to: '/risk', label: '风控管理', icon: ShieldCheck },
  { to: '/sentiment', label: '舆情数据', icon: Newspaper },
  { to: '/dl-models', label: 'DL模型研究', icon: Brain },
  { to: '/portfolio', label: 'Portfolio', icon: Wallet },
  { to: '/logs', label: '系统日志', icon: ScrollText },
  { to: '/chat', label: 'AI Chat', icon: MessageSquare },
];

export default function Sidebar() {
  return (
    <aside className="fixed left-0 top-0 h-screen w-60 z-40 bg-[#1e293b] flex flex-col border-r border-[#334155]">
      <div className="flex items-center gap-2 px-5 py-6">
        <TrendingUp className="h-7 w-7 text-[#3b82f6]" />
        <span className="text-xl font-bold text-[#f8fafc]">QuantTrader</span>
      </div>

      <nav className="flex-1 flex flex-col gap-1 px-3">
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-[#3b82f6]/15 text-[#3b82f6]'
                  : 'text-[#94a3b8] hover:bg-[#334155] hover:text-[#f8fafc]'
              }`
            }
          >
            <Icon className="h-5 w-5" />
            {label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
