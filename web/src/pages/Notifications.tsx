import { useState, useEffect, useCallback } from 'react';
import {
  getNotifications,
  getNotificationConfig,
  saveNotificationConfig,
  markNotificationRead,
  markAllNotificationsRead,
  testNotification,
  type NotificationItem,
  type NotificationConfig,
} from '../api/client';

// ── Event type badge colors ────────────────────────────────────────
const eventColors: Record<string, string> = {
  order_filled: 'bg-green-500/20 text-green-400',
  order_rejected: 'bg-red-500/20 text-red-400',
  risk_alert: 'bg-yellow-500/20 text-yellow-400',
  engine_started: 'bg-blue-500/20 text-blue-400',
  engine_stopped: 'bg-gray-500/20 text-gray-400',
  test: 'bg-purple-500/20 text-purple-400',
};

const eventLabels: Record<string, string> = {
  order_filled: '订单成交',
  order_rejected: '订单拒绝',
  risk_alert: '风控警报',
  engine_started: '引擎启动',
  engine_stopped: '引擎停止',
  test: '测试通知',
};

// ── Default config ─────────────────────────────────────────────────
const defaultConfig: NotificationConfig = {
  enabled: true,
  in_app: true,
  email: { enabled: false, smtp_host: 'smtp.qq.com', smtp_port: 465, username: '', password: '', from: '', to: [], tls: true },
  webhook: { enabled: false, provider: 'dingtalk', url: '', secret: '' },
  events: { order_filled: true, order_rejected: true, risk_alert: true, engine_started: false, engine_stopped: true },
};

// ── Settings Tab ───────────────────────────────────────────────────
function SettingsTab({ config, setConfig, onSave, onTest, testResults, saving, testing }: {
  config: NotificationConfig;
  setConfig: (c: NotificationConfig) => void;
  onSave: () => void;
  onTest: () => void;
  testResults: Array<{ channel: string; success: boolean; message: string }>;
  saving: boolean;
  testing: boolean;
}) {
  const upd = (patch: Partial<NotificationConfig>) => setConfig({ ...config, ...patch });

  return (
    <div className="space-y-6">
      {/* Master Switch */}
      <div className="bg-[#1e293b] rounded-xl p-5 border border-[#334155]">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-[#f8fafc]">🔔 通知总开关</h3>
            <p className="text-sm text-[#94a3b8]">启用后，订单成交等事件将通过配置的渠道发送通知</p>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input type="checkbox" checked={config.enabled} onChange={e => upd({ enabled: e.target.checked })} className="sr-only peer" />
            <div className="w-11 h-6 bg-gray-600 peer-checked:bg-blue-500 rounded-full peer after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:after:translate-x-full" />
          </label>
        </div>
      </div>

      {/* Event Filter */}
      <div className="bg-[#1e293b] rounded-xl p-5 border border-[#334155]">
        <h3 className="text-lg font-semibold text-[#f8fafc] mb-4">📋 通知事件</h3>
        <div className="grid grid-cols-2 gap-3">
          {([
            ['order_filled', '✅ 订单成交'],
            ['order_rejected', '❌ 订单拒绝'],
            ['risk_alert', '⚠️ 风控警报'],
            ['engine_started', '🚀 引擎启动'],
            ['engine_stopped', '🛑 引擎停止'],
          ] as [keyof NotificationConfig['events'], string][]).map(([key, label]) => (
            <label key={key} className="flex items-center gap-2 text-sm text-[#e2e8f0] cursor-pointer">
              <input type="checkbox"
                checked={config.events[key]}
                onChange={e => upd({ events: { ...config.events, [key]: e.target.checked } })}
                className="rounded bg-[#334155] border-[#475569] text-blue-500" />
              {label}
            </label>
          ))}
        </div>
      </div>

      {/* Webhook */}
      <div className="bg-[#1e293b] rounded-xl p-5 border border-[#334155]">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-[#f8fafc]">🌐 Webhook 推送</h3>
          <label className="relative inline-flex items-center cursor-pointer">
            <input type="checkbox" checked={config.webhook.enabled}
              onChange={e => upd({ webhook: { ...config.webhook, enabled: e.target.checked } })}
              className="sr-only peer" />
            <div className="w-11 h-6 bg-gray-600 peer-checked:bg-blue-500 rounded-full peer after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:after:translate-x-full" />
          </label>
        </div>
        {config.webhook.enabled && (
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-[#94a3b8] mb-1">平台</label>
              <select value={config.webhook.provider}
                onChange={e => upd({ webhook: { ...config.webhook, provider: e.target.value } })}
                className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-[#e2e8f0] text-sm">
                <option value="dingtalk">钉钉 (DingTalk)</option>
                <option value="wechat">企业微信 (WeChat Work)</option>
                <option value="slack">Slack</option>
                <option value="custom">自定义 HTTP</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-[#94a3b8] mb-1">Webhook URL</label>
              <input type="url" value={config.webhook.url}
                onChange={e => upd({ webhook: { ...config.webhook, url: e.target.value } })}
                placeholder="https://oapi.dingtalk.com/robot/send?access_token=..."
                className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-[#e2e8f0] text-sm" />
            </div>
            <div>
              <label className="block text-xs text-[#94a3b8] mb-1">Secret (可选)</label>
              <input type="password" value={config.webhook.secret}
                onChange={e => upd({ webhook: { ...config.webhook, secret: e.target.value } })}
                className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-[#e2e8f0] text-sm" />
            </div>
          </div>
        )}
      </div>

      {/* Email */}
      <div className="bg-[#1e293b] rounded-xl p-5 border border-[#334155]">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-[#f8fafc]">📧 邮件通知</h3>
          <label className="relative inline-flex items-center cursor-pointer">
            <input type="checkbox" checked={config.email.enabled}
              onChange={e => upd({ email: { ...config.email, enabled: e.target.checked } })}
              className="sr-only peer" />
            <div className="w-11 h-6 bg-gray-600 peer-checked:bg-blue-500 rounded-full peer after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:after:translate-x-full" />
          </label>
        </div>
        {config.email.enabled && (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-[#94a3b8] mb-1">SMTP 服务器</label>
                <input type="text" value={config.email.smtp_host}
                  onChange={e => upd({ email: { ...config.email, smtp_host: e.target.value } })}
                  placeholder="smtp.qq.com"
                  className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-[#e2e8f0] text-sm" />
              </div>
              <div>
                <label className="block text-xs text-[#94a3b8] mb-1">端口</label>
                <input type="number" value={config.email.smtp_port}
                  onChange={e => upd({ email: { ...config.email, smtp_port: parseInt(e.target.value) || 465 } })}
                  className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-[#e2e8f0] text-sm" />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-[#94a3b8] mb-1">用户名</label>
                <input type="text" value={config.email.username}
                  onChange={e => upd({ email: { ...config.email, username: e.target.value } })}
                  className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-[#e2e8f0] text-sm" />
              </div>
              <div>
                <label className="block text-xs text-[#94a3b8] mb-1">密码 / 授权码</label>
                <input type="password" value={config.email.password}
                  onChange={e => upd({ email: { ...config.email, password: e.target.value } })}
                  className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-[#e2e8f0] text-sm" />
              </div>
            </div>
            <div>
              <label className="block text-xs text-[#94a3b8] mb-1">发件人地址</label>
              <input type="email" value={config.email.from}
                onChange={e => upd({ email: { ...config.email, from: e.target.value } })}
                className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-[#e2e8f0] text-sm" />
            </div>
            <div>
              <label className="block text-xs text-[#94a3b8] mb-1">收件人 (逗号分隔)</label>
              <input type="text" value={config.email.to.join(', ')}
                onChange={e => upd({ email: { ...config.email, to: e.target.value.split(',').map(s => s.trim()).filter(Boolean) } })}
                placeholder="user@example.com, admin@example.com"
                className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-[#e2e8f0] text-sm" />
            </div>
            <label className="flex items-center gap-2 text-sm text-[#e2e8f0] cursor-pointer">
              <input type="checkbox" checked={config.email.tls}
                onChange={e => upd({ email: { ...config.email, tls: e.target.checked } })}
                className="rounded bg-[#334155] border-[#475569] text-blue-500" />
              启用 TLS 加密
            </label>
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="flex gap-3">
        <button onClick={onSave} disabled={saving}
          className="px-6 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-medium disabled:opacity-50">
          {saving ? '保存中...' : '💾 保存配置'}
        </button>
        <button onClick={onTest} disabled={testing}
          className="px-6 py-2 bg-[#334155] hover:bg-[#475569] text-[#e2e8f0] rounded-lg font-medium disabled:opacity-50">
          {testing ? '发送中...' : '🔔 发送测试通知'}
        </button>
      </div>

      {/* Test Results */}
      {testResults.length > 0 && (
        <div className="bg-[#1e293b] rounded-xl p-4 border border-[#334155]">
          <h4 className="text-sm font-semibold text-[#f8fafc] mb-2">测试结果</h4>
          {testResults.map((r, i) => (
            <div key={i} className="flex items-center gap-2 text-sm py-1">
              <span className={r.success ? 'text-green-400' : 'text-red-400'}>
                {r.success ? '✅' : '❌'}
              </span>
              <span className="text-[#e2e8f0] font-mono">{r.channel}</span>
              <span className="text-[#94a3b8]">{r.message}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── History Tab ────────────────────────────────────────────────────
function HistoryTab({ notifications, onMarkRead, onMarkAllRead, onRefresh }: {
  notifications: NotificationItem[];
  onMarkRead: (id: string) => void;
  onMarkAllRead: () => void;
  onRefresh: () => void;
}) {
  const unreadCount = notifications.filter(n => !n.read).length;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="text-sm text-[#94a3b8]">
          共 {notifications.length} 条通知，{unreadCount} 条未读
        </div>
        <div className="flex gap-2">
          <button onClick={onRefresh}
            className="px-3 py-1.5 text-sm bg-[#334155] hover:bg-[#475569] text-[#e2e8f0] rounded-lg">
            🔄 刷新
          </button>
          {unreadCount > 0 && (
            <button onClick={onMarkAllRead}
              className="px-3 py-1.5 text-sm bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 rounded-lg">
              ✓ 全部已读
            </button>
          )}
        </div>
      </div>

      {notifications.length === 0 ? (
        <div className="text-center py-16 text-[#94a3b8]">
          <p className="text-4xl mb-3">🔔</p>
          <p>暂无通知</p>
        </div>
      ) : (
        <div className="space-y-2">
          {notifications.map(n => (
            <div key={n.id}
              className={`bg-[#1e293b] rounded-xl p-4 border transition-colors cursor-pointer
                ${n.read ? 'border-[#334155] opacity-70' : 'border-blue-500/30 bg-blue-500/5'}`}
              onClick={() => !n.read && onMarkRead(n.id)}>
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${eventColors[n.event_type] || 'bg-gray-500/20 text-gray-400'}`}>
                      {eventLabels[n.event_type] || n.event_type}
                    </span>
                    {!n.read && <span className="w-2 h-2 rounded-full bg-blue-500" />}
                    <span className="text-xs text-[#64748b]">{n.timestamp}</span>
                  </div>
                  <p className="text-sm font-medium text-[#f8fafc]">{n.title}</p>
                  <p className="text-xs text-[#94a3b8] mt-1 whitespace-pre-line">{n.message}</p>
                </div>
                <div className="text-xs text-[#64748b] whitespace-nowrap">
                  {n.channels.split(',').map(ch => (
                    <span key={ch} className="inline-block px-1.5 py-0.5 bg-[#0f172a] rounded mr-1">{ch}</span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────────────
export default function Notifications() {
  const [tab, setTab] = useState<'history' | 'settings'>('history');
  const [notifications, setNotifications] = useState<NotificationItem[]>([]);
  const [config, setConfig] = useState<NotificationConfig>(defaultConfig);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResults, setTestResults] = useState<Array<{ channel: string; success: boolean; message: string }>>([]);

  const loadNotifications = useCallback(async () => {
    try {
      const data = await getNotifications(100);
      setNotifications(data.notifications || []);
    } catch { /* ignore */ }
  }, []);

  const loadConfig = useCallback(async () => {
    try {
      const cfg = await getNotificationConfig();
      setConfig(cfg);
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    loadNotifications();
    loadConfig();
  }, [loadNotifications, loadConfig]);

  const handleSave = async () => {
    setSaving(true);
    try {
      await saveNotificationConfig(config);
    } finally {
      setSaving(false);
    }
  };

  const handleTest = async () => {
    setTesting(true);
    try {
      const res = await testNotification();
      setTestResults(res.results || []);
      loadNotifications();
    } finally {
      setTesting(false);
    }
  };

  const handleMarkRead = async (id: string) => {
    await markNotificationRead(id);
    setNotifications(prev => prev.map(n => n.id === id ? { ...n, read: true } : n));
  };

  const handleMarkAllRead = async () => {
    await markAllNotificationsRead();
    setNotifications(prev => prev.map(n => ({ ...n, read: true })));
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold text-[#f8fafc] mb-6">🔔 通知中心</h1>

      {/* Tabs */}
      <div className="flex gap-1 bg-[#0f172a] rounded-lg p-1 mb-6 w-fit">
        {[
          { id: 'history' as const, label: '📨 通知记录' },
          { id: 'settings' as const, label: '⚙️ 通知设置' },
        ].map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors
              ${tab === t.id ? 'bg-[#334155] text-[#f8fafc]' : 'text-[#94a3b8] hover:text-[#e2e8f0]'}`}>
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'history' ? (
        <HistoryTab
          notifications={notifications}
          onMarkRead={handleMarkRead}
          onMarkAllRead={handleMarkAllRead}
          onRefresh={loadNotifications}
        />
      ) : (
        <SettingsTab
          config={config}
          setConfig={setConfig}
          onSave={handleSave}
          onTest={handleTest}
          testResults={testResults}
          saving={saving}
          testing={testing}
        />
      )}
    </div>
  );
}
