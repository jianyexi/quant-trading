export interface KlineData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface StockQuote {
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
  volume: number;
  timestamp: string;
}

export interface BacktestResult {
  id: string;
  strategy: string;
  symbol: string;
  start: string;
  end: string;
  initial_capital: number;
  final_value: number;
  total_return_percent: number;
  sharpe_ratio: number;
  max_drawdown_percent: number;
  win_rate_percent: number;
  total_trades: number;
  profit_factor: number;
  status: string;
}

export interface Position {
  symbol: string;
  name: string;
  shares: number;
  avg_cost: number;
  current_price: number;
  pnl: number;
}

export interface Portfolio {
  total_value: number;
  cash: number;
  total_pnl: number;
  positions: Position[];
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: Date;
}

export interface StrategyConfig {
  name: string;
  displayName: string;
  description: string;
  parameters: StrategyParam[];
}

export interface StrategyParam {
  key: string;
  label: string;
  type: 'number' | 'select';
  default: number | string;
  min?: number;
  max?: number;
  step?: number;
  options?: { value: string; label: string }[];
}
