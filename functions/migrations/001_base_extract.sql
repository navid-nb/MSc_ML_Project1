-- Center for Research in Security Prices (CRSP) daily stock file (dsf)
SELECT d.cusip,                                              -- 8-digit CUSIP identifier
       d.permno,                                             -- CRSP permanent security number (unique ID)
       s.ticker,                                             -- ticker
       "date",                                               -- trading date (YYYY-MM-DD)
       bidlo,                                                -- daily low (bid/low price)
       askhi,                                                -- daily high (ask/high price)
       prc,                                                  -- raw closing price (can be bid/ask avg, negative if bid)
       vol,                                                  -- trading volume (raw, unadjusted)
       ret,                                                  -- daily return (with dividends, adjusted for splits)
       bid,                                                  -- bid price
       ask,                                                  -- ask price
       shrout,                                               -- shares outstanding (in thousands, raw)
       cfacpr,                                               -- cumulative factor to adjust prices
       cfacshr,                                              -- cumulative factor to adjust shares/volume
       openprc,                                              -- opening price
       numtrd,                                               -- number of trades (Nasdaq only)
       retx,                                                 -- daily return (excluding dividends)
       -- manual derivations
       abs(prc) / NULLIF(cfacpr, 0)                         AS adj_prc,         -- adjusted close price (split/dividend adjusted)
       abs(bidlo) / NULLIF(cfacpr, 0)                       AS adj_bidlo,       -- adjusted bid/low price
       abs(askhi) / NULLIF(cfacpr, 0)                       AS adj_askhi,       -- adjusted ask/high price
       abs(openprc) / NULLIF(cfacpr, 0)                     AS adj_openprc,     -- adjusted open price
       vol * cfacshr                                        AS adj_vol,         -- adjusted volume
       shrout * cfacshr                                     AS adj_shrout,      -- adjusted shares outstanding (useful for adj. market cap)
       abs(prc) / NULLIF(cfacpr, 0) * (shrout * cfacshr)    AS adj_mktcap       -- adjusted market cap
FROM crsp.dsf d
JOIN crsp.stocknames s
  ON d.permno = s.permno
 AND d."date" >= s.namedt
 AND d."date" <= COALESCE(s.nameenddt, DATE '9999-12-31')
WHERE d."date" >= %(start)s::date
  AND d."date" <  %(end)s::date
  AND s.ticker IN (
    'AAPL', 'NVDA', 'MSFT', 'AMZN', 'TSLA', 'GOOGL', 'LLY', 'WMT', 'JPM', 'BRK-B',
    'V', 'MA', 'XOM', 'ORCL', 'UNH', 'COST', 'PG', 'HD', 'NFLX',
    'JNJ', 'BAC', 'CRM', 'QQQ', 'ABBV', 'KO', 'CVX', 'TMUS', 'MRK', 'CSCO',
    'WFC', 'ACN', 'NOW', 'TSM', 'AXP', 'PEP', 'MCD', 'IBM', 'MS', 'DIS',
    'TMO', 'ABT', 'AMD', 'ADBE', 'PM', 'ISRG', 'GE', 'GS', 'INTU', 'CAT',
    'TXN', 'QCOM', 'RY', 'VZ', 'DHR', 'BKNG', 'T', 'BLK', 'SPGI',
    'RTX', 'PFE', 'NEE', 'HON', 'CMCSA', 'PGR', 'AMGN', 'LOW', 'ANET', 'UNP',
    'SYK', 'TJX', 'C', 'BA', 'SCHW', 'BSX', 'KKR', 'ETN',
    'COP', 'BX', 'PANW', 'ADP'
  )
;