SELECT
  /* --- identifiers --- */
  gvkey,
  datadate:,
  fyearq,
  fqtr,
  tic,
  cusip,
  conm,

  cshtrq     AS shs_traded_q,  -- Shares traded in quarter (proxy for volume)
  cshoq      AS shs_out_q,     -- Common shares outstanding
  mkvaltq    AS mkt_value_q,   -- Market value (total)
  dvpsxq     AS div_ps_exdate_q, -- Dividends per share (ex-date, quarter)
  adjex      AS adj_factor_ex, -- Cumulative adjustment factor by ex-date

  /* --- income statement (valuation, margins, coverage, growth) --- */
  saleq      AS sales_q,       -- Sales/Turnover (net)
  revtq      AS revenue_q,     -- Revenue - Total
  cogsq      AS cogs_q,        -- Cost of Goods Sold
  xsgaq      AS sga_q,         -- SG&A (for F-score / quality if used later)
  xrdq       AS rd_exp_q,      -- R&D (optional for quality/PEG adjustments)
  oibdpq     AS ebitda_q,      -- Operating Income Before Depreciation (EBITDA)
  oiadpq     AS ebit_q,        -- Operating Income After Depreciation (EBIT)
  xintq      AS int_exp_q,     -- Interest expense (for coverage)
  niq        AS net_income_q,  -- Net income (for P/E, ROE, payouts)
  epspxq     AS eps_basic_excl_xo_q, -- EPS basic excl. extraordinary (for P/E, payout)

  /* --- balance sheet (book, liquidity, leverage, Z/F-score) --- */
  atq        AS assets_total_q,
  ltq        AS liab_total_q,
  ceqq       AS common_equity_q,   -- Book equity (for P/B, ROE)
  seqq       AS sh_equity_total_q, -- Total equity (alternative book)
  actq       AS cur_assets_q,      -- Current assets
  lctq       AS cur_liab_q,        -- Current liabilities
  cheq       AS cash_sti_q,        -- Cash & short-term investments
  invtq      AS inventory_q,       -- Inventory (quick ratio, inv. turnover)
  rectq      AS receivables_q,     -- Receivables (A/R turnover)
  ppentq     AS ppe_net_q,         -- PPE (Z-score component)
  wcapq      AS working_cap_q,     -- Working capital (Z-score component)
  req        AS retained_earn_q,   -- Retained earnings (Z-score; F-score)

  /* --- capital structure (EV, leverage, net debt / EBITDA) --- */
  dlttq      AS debt_lt_q,      -- Long-term debt
  dlcq       AS debt_curr_q,    -- Current debt
  npq        AS notes_pay_q,    -- Notes payable (optional ST debt detail)

  /* --- payouts / buybacks (payout ratio, net payout yield) --- */
  cshopq     AS shs_repurchased_q, -- Total shares repurchased (quarter)
  prcraq     AS avg_repurchase_px_q -- Avg repurchase price per share (quarter)
FROM comp.fundq
WHERE datadate >= %(start)s::date
  AND datadate <  %(end)s::date
  -- (optional hygiene you can uncomment later)
  -- AND finalq = 'Y' AND consol = 'C' AND datafmt = 'STD'
ORDER BY gvkey, datadate;