import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { MarketProvider } from './contexts/MarketContext';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import MarketData from './pages/MarketData';
import StrategyConfig from './pages/StrategyConfig';
import Portfolio from './pages/Portfolio';
import Chat from './pages/Chat';
import Screener from './pages/Screener';
import AutoTrade from './pages/AutoTrade';
import Sentiment from './pages/Sentiment';
import DLModels from './pages/DLModels';
import LLMTraining from './pages/LLMTraining';
import RiskManagement from './pages/RiskManagement';
import Logs from './pages/Logs';
import Notifications from './pages/Notifications';
import Pipeline from './pages/Pipeline';
import History from './pages/History';
import Services from './pages/Services';
import Backtest from './pages/Backtest';
import FactorMining from './pages/factor-mining';
import DataQuality from './pages/DataQuality';

export default function App() {
  return (
    <MarketProvider>
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="market" element={<MarketData />} />
          <Route path="strategy" element={<StrategyConfig />} />
          <Route path="pipeline" element={<Pipeline />} />
          <Route path="history" element={<History />} />
          <Route path="screener" element={<Screener />} />
          <Route path="autotrade" element={<AutoTrade />} />
          <Route path="risk" element={<RiskManagement />} />
          <Route path="sentiment" element={<Sentiment />} />
          <Route path="dl-models" element={<DLModels />} />
          <Route path="llm-training" element={<LLMTraining />} />
          <Route path="portfolio" element={<Portfolio />} />
          <Route path="notifications" element={<Notifications />} />
          <Route path="logs" element={<Logs />} />
          <Route path="metrics" element={<Navigate to="/" replace />} />
          <Route path="reports" element={<Navigate to="/" replace />} />
          <Route path="latency" element={<Navigate to="/" replace />} />
          <Route path="backtest" element={<Backtest />} />
          <Route path="factor-mining" element={<FactorMining />} />
          <Route path="data-quality" element={<DataQuality />} />
          <Route path="services" element={<Services />} />
          <Route path="chat" element={<Chat />} />
        </Route>
      </Routes>
    </BrowserRouter>
    </MarketProvider>
  );
}
