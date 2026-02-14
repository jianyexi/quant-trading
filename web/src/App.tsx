import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import MarketData from './pages/MarketData';
import StrategyConfig from './pages/StrategyConfig';
import Backtest from './pages/Backtest';
import Portfolio from './pages/Portfolio';
import Chat from './pages/Chat';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="market" element={<MarketData />} />
          <Route path="strategy" element={<StrategyConfig />} />
          <Route path="backtest" element={<Backtest />} />
          <Route path="portfolio" element={<Portfolio />} />
          <Route path="chat" element={<Chat />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
