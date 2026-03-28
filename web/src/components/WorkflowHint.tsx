import React from 'react';

interface Step {
  label: string;
  path: string;
  icon: string;
}

const WORKFLOW_STEPS: Step[] = [
  { label: '数据获取', path: '/market', icon: '📡' },
  { label: '因子挖掘', path: '/factor-mining', icon: '🔬' },
  { label: '策略配置', path: '/strategy', icon: '⚙️' },
  { label: '回测验证', path: '/backtest', icon: '📊' },
  { label: '纸上交易', path: '/autotrade', icon: '📝' },
  { label: '实盘监控', path: '/', icon: '🖥️' },
];

interface Props {
  currentPath: string;
}

export default function WorkflowHint({ currentPath }: Props) {
  const currentIndex = WORKFLOW_STEPS.findIndex(s => s.path === currentPath);
  if (currentIndex < 0) return null;

  return (
    <div className="mt-8 pt-4 border-t border-gray-700/50">
      <div className="flex items-center justify-center gap-1 text-sm">
        {WORKFLOW_STEPS.map((step, i) => {
          const isCurrent = i === currentIndex;
          const isPast = i < currentIndex;
          const isNext = i === currentIndex + 1;

          return (
            <React.Fragment key={step.path}>
              {i > 0 && <span className="text-gray-600 mx-1">→</span>}
              <a
                href={step.path}
                className={`px-2 py-1 rounded transition-colors ${
                  isCurrent
                    ? 'bg-blue-600/30 text-blue-300 font-semibold border border-blue-500/50'
                    : isNext
                    ? 'bg-green-900/30 text-green-400 hover:bg-green-800/40 border border-green-600/30 animate-pulse'
                    : isPast
                    ? 'text-gray-500 hover:text-gray-400'
                    : 'text-gray-600 hover:text-gray-500'
                }`}
              >
                {step.icon} {step.label}
              </a>
            </React.Fragment>
          );
        })}
      </div>
      {currentIndex < WORKFLOW_STEPS.length - 1 && (
        <p className="text-center text-xs text-gray-500 mt-2">
          下一步: <a href={WORKFLOW_STEPS[currentIndex + 1].path} className="text-green-400 hover:underline">
            {WORKFLOW_STEPS[currentIndex + 1].icon} {WORKFLOW_STEPS[currentIndex + 1].label} →
          </a>
        </p>
      )}
    </div>
  );
}
