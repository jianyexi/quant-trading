import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import MarketData from './pages/MarketData';
import StrategyConfig from './pages/StrategyConfig';
import Backtest from './pages/Backtest';
import Portfolio from './pages/Portfolio';
import Chat from './pages/Chat';
import Screener from './pages/Screener';
import AutoTrade from './pages/AutoTrade';
import Sentiment from './pages/Sentiment';
import DLModels from './pages/DLModels';
import RiskManagement from './pages/RiskManagement';
import Logs from './pages/Logs';
import FactorMining from './pages/FactorMining';
import Notifications from './pages/Notifications';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="market" element={<MarketData />} />
          <Route path="strategy" element={<StrategyConfig />} />
          <Route path="backtest" element={<Backtest />} />
          <Route path="screener" element={<Screener />} />
          <Route path="autotrade" element={<AutoTrade />} />
          <Route path="risk" element={<RiskManagement />} />
          <Route path="sentiment" element={<Sentiment />} />
          <Route path="dl-models" element={<DLModels />} />
          <Route path="factor-mining" element={<FactorMining />} />
          <Route path="portfolio" element={<Portfolio />} />
          <Route path="notifications" element={<Notifications />} />
          <Route path="logs" element={<Logs />} />
          <Route path="chat" element={<Chat />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
