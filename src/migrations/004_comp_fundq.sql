-- Compustat Quarterly (FUNDQ)
SELECT f.gvkey,                          -- firm identifier (stable)
       f.datadate,                       -- fiscal quarter period end (accounting date)
       f.fyearq,                         -- fiscal year
       f.fqtr,                           -- fiscal quarter (1..4)
       f.tic,                            -- ticker (changes possible; readability)
       f.cusip,                          -- CUSIP (mapping; not unique/stable over time)
       f.conm,                           -- company name (human-readable)
       f.iid,                            -- issue ID (security-level within gvkey)
       f.rdq,                            -- earnings announcement date (event timing / PEAD)

       -- Market / price features
       f.prccq   AS px_close_q,          -- quarter-end close price
       f.prchq   AS px_high_q,           -- quarter high
       f.prclq   AS px_low_q,            -- quarter low
       f.mkvaltq AS mkt_value_q,         -- total market value (units may vary by install)
       f.adjex   AS adj_factor_ex,       -- cumulative adjustment factor by ex-date
       f.dvpsxq  AS div_ps_exdate_q,     -- dividends/share (ex-date) for dividend yield / payout

       -- Trading activity / size
       f.cshtrq  AS shs_traded_q,        -- shares traded in quarter (volume proxy)
       f.cshoq   AS shs_out_q,           -- common shares outstanding (quarter end)
       CASE
           WHEN f.cshoq IS NOT NULL AND f.cshoq <> 0
               THEN f.cshtrq / f.cshoq
           END   AS turnover_q,          -- share turnover proxy

       -- Income statement (profitability, margins, growth)
       f.saleq   AS sales_q,             -- sales (net)
       f.revtq   AS revenue_q,           -- revenue (total)
       f.cogsq   AS cogs_q,              -- cost of goods sold
       f.xsgaq   AS sga_q,               -- SG&A (quality/F-score inputs)
       f.xrdq    AS rd_exp_q,            -- R&D expense (innovation/mispricing proxies)
       f.oibdpq  AS ebitda_q,            -- operating income before depreciation
       f.oiadpq  AS ebit_q,              -- operating income after depreciation
       f.xintq   AS int_exp_q,           -- interest expense (coverage)
       f.niq     AS net_income_q,        -- net income
       f.epspxq  AS eps_basic_excl_xo_q, -- EPS (basic, excl. extraordinary)

       -- Balance sheet (book, liquidity, leverage, Z/F-score inputs)
       f.atq     AS assets_total_q,      -- total assets
       f.ltq     AS liab_total_q,        -- total liabilities
       f.ceqq    AS common_equity_q,     -- common/ordinary equity (book)
       f.seqq    AS sh_equity_total_q,   -- total shareholders’ equity
       f.actq    AS cur_assets_q,        -- current assets
       f.lctq    AS cur_liab_q,          -- current liabilities
       f.cheq    AS cash_sti_q,          -- cash & short-term investments
       f.invtq   AS inventory_q,         -- inventory (turnover/quality)
       f.rectq   AS receivables_q,       -- receivables (accruals/turnover)
       f.ppentq  AS ppe_net_q,           -- property, plant & equipment (net)
       f.wcapq   AS working_cap_q,       -- working capital
       f.req     AS retained_earn_q,     -- retained earnings (profitability persistence)

       -- Capital structure (leverage / financing frictions)
       f.dlttq   AS debt_lt_q,           -- long-term debt
       f.dlcq    AS debt_curr_q,         -- current (short-term) debt
       f.npq     AS notes_pay_q,         -- notes payable (extra ST debt detail)

       -- Payouts / buybacks (net payout yield, equity issuance)
       f.cshopq  AS shs_repurchased_q,   -- total shares repurchased (quarter)
       f.prcraq  AS avg_repurchase_px_q  -- average repurchase price per share (quarter)

FROM comp.fundq AS f
WHERE f.datadate >= %(start)s::date
  AND f.datadate <  %(end)s::date
    AND f.finalq = 'Y'          -- keep finalized filings
    AND f.consol = 'C'          -- consolidated
    AND f.datafmt = 'STD'       -- standard format
ORDER BY f.gvkey, f.datadate;