import { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
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
  Dna,
  Bell,
  Activity,
  FileBarChart,
  Timer,
  Workflow,
  History,
  ChevronDown,
  ChevronRight,
  type LucideIcon,
} from 'lucide-react';

interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
}

interface NavGroup {
  label: string;
  icon: LucideIcon;
  items: NavItem[];
}

type NavEntry = NavItem | NavGroup;

function isGroup(entry: NavEntry): entry is NavGroup {
  return 'items' in entry;
}

const navigation: NavEntry[] = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },

  // 量化研究
  {
    label: '量化研究',
    icon: Dna,
    items: [
      { to: '/pipeline', label: '量化流水线', icon: Workflow },
      { to: '/factor-mining', label: '因子挖掘', icon: Dna },
      { to: '/dl-models', label: 'DL模型研究', icon: Brain },
      { to: '/backtest', label: '策略回测', icon: FlaskConical },
      { to: '/history', label: '任务历史', icon: History },
    ],
  },

  // 交易执行
  {
    label: '交易执行',
    icon: TrendingUp,
    items: [
      { to: '/market', label: '行情数据', icon: TrendingUp },
      { to: '/screener', label: '智能选股', icon: Search },
      { to: '/strategy', label: '策略管理', icon: Settings },
      { to: '/autotrade', label: '自动交易', icon: Bot },
      { to: '/portfolio', label: '持仓管理', icon: Wallet },
      { to: '/risk', label: '风控管理', icon: ShieldCheck },
    ],
  },

  // 数据 & 监控
  {
    label: '数据 & 监控',
    icon: Activity,
    items: [
      { to: '/sentiment', label: '舆情数据', icon: Newspaper },
      { to: '/notifications', label: '通知中心', icon: Bell },
      { to: '/logs', label: '系统日志', icon: ScrollText },
      { to: '/metrics', label: '性能监控', icon: Activity },
      { to: '/reports', label: '统计报表', icon: FileBarChart },
      { to: '/latency', label: '延迟分析', icon: Timer },
    ],
  },

  { to: '/chat', label: 'AI Chat', icon: MessageSquare },
];

const linkCls = (isActive: boolean) =>
  `flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
    isActive
      ? 'bg-[#3b82f6]/15 text-[#3b82f6]'
      : 'text-[#94a3b8] hover:bg-[#334155] hover:text-[#f8fafc]'
  }`;

function NavGroupSection({ group }: { group: NavGroup }) {
  const location = useLocation();
  const hasActive = group.items.some(item => location.pathname === item.to);
  const [open, setOpen] = useState(hasActive);

  return (
    <div>
      <button
        onClick={() => setOpen(o => !o)}
        className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-[#cbd5e1] hover:bg-[#334155] hover:text-[#f8fafc] transition-colors"
      >
        <group.icon className="h-4 w-4 text-[#64748b]" />
        <span className="flex-1 text-left">{group.label}</span>
        {open
          ? <ChevronDown className="h-3.5 w-3.5 text-[#64748b]" />
          : <ChevronRight className="h-3.5 w-3.5 text-[#64748b]" />
        }
      </button>
      {open && (
        <div className="ml-3 mt-0.5 flex flex-col gap-0.5 border-l border-[#334155] pl-2">
          {group.items.map(({ to, label, icon: Icon }) => (
            <NavLink key={to} to={to} end={to === '/'} className={({ isActive }) => linkCls(isActive)}>
              <Icon className="h-4 w-4" />
              {label}
            </NavLink>
          ))}
        </div>
      )}
    </div>
  );
}

export default function Sidebar() {
  return (
    <aside className="fixed left-0 top-0 h-screen w-60 z-40 bg-[#1e293b] flex flex-col border-r border-[#334155]">
      <div className="flex items-center gap-2 px-5 py-5">
        <TrendingUp className="h-7 w-7 text-[#3b82f6]" />
        <span className="text-xl font-bold text-[#f8fafc]">QuantTrader</span>
      </div>

      <nav className="flex-1 flex flex-col gap-0.5 px-3 overflow-y-auto">
        {navigation.map((entry) =>
          isGroup(entry) ? (
            <NavGroupSection key={entry.label} group={entry} />
          ) : (
            <NavLink key={entry.to} to={entry.to} end={entry.to === '/'} className={({ isActive }) => linkCls(isActive)}>
              <entry.icon className="h-5 w-5" />
              {entry.label}
            </NavLink>
          )
        )}
      </nav>
    </aside>
  );
}
