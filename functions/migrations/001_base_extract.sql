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
 'NVDA', 'AAPL', 'AMZN', 'JPM', 'WMT', 'LLY', 'BRK-B', 
 'UNH', 'NFLX', 'XOM', 'ORCL', 'COST', 'PG', 'HD', 'JNJ', 'BAC', 
 'CRM', 'KO', 'CVX', 'MRK', 'CSCO', 'WFC', 'ACN', 'AXP', 'PEP', 
 'MCD', 'IBM', 'DIS', 'TMO', 'ABT', 'AMD', 'ADBE', 'ISRG', 
 'GE', 'GS', 'INTU', 'CAT', 'TXN', 'QCOM', 'RY', 'VZ', 'DHR',
 'BLK', 'PFE', 'HON', 'CMCSA', 
 'PGR', 'AMGN', 'LOW', 'UNP', 'SYK', 'TJX', 'C', 'BA', 
 'BSX', 'ETN', 'COP', 'ADP', 'SBUX', 'VRTX', 'GILD', 'ADI', 
 'LRCX', 'DE', 'SO', 'MU', 'PLD', 'REGN', 'DUK', 
 'SHW', 'KLAC', 'CI', 'BMY', 'APH', 'MCO', 'ROST', 'AMAT', 'MCK', 'EOG', 
 'PH', 'ORLY', 'GD', 'NOC', 'MSI', 
 'ITW', 'TT', 'PCAR', 'AMT', 'UL', 'BHP', 'TD', 'COF', 'LOW', 'HON', 'UBS', 
 'YUM', 'RCL', 'RSG', 'NKE', 'NEM', 'CDNS', 'MSTR'
  )
;