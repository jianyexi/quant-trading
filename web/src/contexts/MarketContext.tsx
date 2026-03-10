import { createContext, useContext, useState, useEffect, type ReactNode } from 'react';

export type MarketRegion = 'ALL' | 'CN' | 'US' | 'HK';

interface MarketContextType {
  market: MarketRegion;
  setMarket: (m: MarketRegion) => void;
  extractMarket: (symbol: string) => 'CN' | 'US' | 'HK';
  filterByMarket: <T>(items: T[], symbolKey: keyof T) => T[];
}

const MarketContext = createContext<MarketContextType | null>(null);

const STORAGE_KEY = 'quant_selected_market';

export function extractMarketFromSymbol(symbol: string): 'CN' | 'US' | 'HK' {
  const s = symbol.toUpperCase();
  if (s.endsWith('.SH') || s.endsWith('.SZ')) return 'CN';
  if (s.endsWith('.HK')) return 'HK';
  // Numeric-only codes (e.g. 600519, 000001) are CN
  if (/^\d{6}$/.test(s)) return 'CN';
  // Everything else (AAPL, GOOGL, BRK.B) is US
  return 'US';
}

export function MarketProvider({ children }: { children: ReactNode }) {
  const [market, setMarketState] = useState<MarketRegion>(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    return (saved as MarketRegion) || 'ALL';
  });

  const setMarket = (m: MarketRegion) => {
    setMarketState(m);
    localStorage.setItem(STORAGE_KEY, m);
  };

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved && saved !== market) setMarketState(saved as MarketRegion);
  }, []);

  const filterByMarket = <T,>(items: T[], symbolKey: keyof T): T[] => {
    if (market === 'ALL') return items;
    return items.filter(item => {
      const sym = String(item[symbolKey] ?? '');
      return extractMarketFromSymbol(sym) === market;
    });
  };

  return (
    <MarketContext.Provider value={{ market, setMarket, extractMarket: extractMarketFromSymbol, filterByMarket }}>
      {children}
    </MarketContext.Provider>
  );
}

export function useMarket(): MarketContextType {
  const ctx = useContext(MarketContext);
  if (!ctx) throw new Error('useMarket must be used within MarketProvider');
  return ctx;
}

export const MARKET_OPTIONS: { value: MarketRegion; label: string; flag: string }[] = [
  { value: 'ALL', label: '全部市场', flag: '🌐' },
  { value: 'CN', label: 'A股', flag: '🇨🇳' },
  { value: 'US', label: '美股', flag: '🇺🇸' },
  { value: 'HK', label: '港股', flag: '🇭🇰' },
];
