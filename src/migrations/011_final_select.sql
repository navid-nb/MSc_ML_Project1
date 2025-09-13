/* 011_final_select.sql
   Tighten early with ticker filters inside CTEs to prevent row explosion.
   Placeholders:
     {{TICKER_FILTER}}  -> e.g., n.ticker IN ('AAPL','MSFT')
   The driver will also inject a per-year date clause at the marker: --__DATE_FILTER__
*/
WITH
names_filt AS (
    -- Only the tickers we care about
    SELECT
        n.permno,
        n.ticker,
        n.ncusip,
        n.namedt,
        n.nameenddt
    FROM stocknames n
    WHERE ({{TICKER_FILTER}})
),
base AS (
    -- CRSP daily spine restricted to our tickers (via name history banding)
    SELECT
        dsf.permno,
        dsf."date",
        dsf.ret,
        dsf.vol,
        dsf.prc,
        dsf.shrout,
        dsf.cfacpr -- cumulative factor to adjust prices for splits and dividends
    FROM dsf
    JOIN names_filt n
      ON dsf.permno = n.permno
     AND dsf."date" BETWEEN n.namedt AND COALESCE(n.nameenddt, DATE '9999-12-31')
),
gvkeys_filt AS (
    -- GVKEY universe reachable from our tickers via COMP.SECM (tic or cusip)
    SELECT DISTINCT s.gvkey
    FROM secm s
    JOIN names_filt n
      ON s.tic  = n.ticker
      OR s.cusip = n.ncusip
),
raw_links AS (
    -- Link PERMNO/DATE -> GVKEY using SEC M snapshots, preferring CUSIP over TICKER
    -- Priority 1: CUSIP snapshot match
    SELECT
        b.permno,
        b."date",
        s.gvkey,
        s.datadate AS snap_dt,
        1 AS priority,
        'CUSIP' AS match_type
    FROM base b
    JOIN names_filt n
      ON b.permno = n.permno
     AND b."date" BETWEEN n.namedt AND COALESCE(n.nameenddt, DATE '9999-12-31')
    JOIN secm s
      ON n.ncusip = s.cusip
     AND s.datadate <= b."date"

    UNION ALL

    -- Priority 2: TICKER snapshot match
    SELECT
        b.permno,
        b."date",
        s.gvkey,
        s.datadate AS snap_dt,
        2 AS priority,
        'TICKER' AS match_type
    FROM base b
    JOIN names_filt n
      ON b.permno = n.permno
     AND b."date" BETWEEN n.namedt AND COALESCE(n.nameenddt, DATE '9999-12-31')
    JOIN secm s
      ON n.ticker = s.tic
     AND s.datadate <= b."date"
),
links AS (
    -- One best link per (permno, date)
    SELECT *
    FROM (
        SELECT
            rl.*,
            ROW_NUMBER() OVER (
                PARTITION BY rl.permno, rl."date"
                ORDER BY rl.priority ASC, rl.snap_dt DESC, rl.gvkey
            ) AS rn
        FROM raw_links rl
    ) z
    WHERE rn = 1
),
comp_fund AS (
    -- Filter Compustat fundamentals down to only relevant GVKEYs
    SELECT
        gvkey,
        datadate,
        fyearq,
        fqtr,
        tic,
        cusip,
        conm,
        shs_traded_q,
        shs_out_q,
        mkt_value_q,
        div_ps_exdate_q,
        adj_factor_ex,
        sales_q,
        revenue_q,
        cogs_q,
        sga_q,
        rd_exp_q,
        ebitda_q,
        ebit_q,
        int_exp_q,
        net_income_q,
        eps_basic_excl_xo_q,
        assets_total_q,
        liab_total_q,
        common_equity_q,
        sh_equity_total_q,
        cur_assets_q,
        cur_liab_q,
        cash_sti_q,
        inventory_q,
        receivables_q,
        ppe_net_q,
        working_cap_q,
        retained_earn_q,
        debt_lt_q,
        debt_curr_q,
        notes_pay_q,
        shs_repurchased_q,
        avg_repurchase_px_q
    FROM fundq q
    WHERE q.gvkey IN (SELECT gvkey FROM gvkeys_filt)
),
ibes_cons_latest AS (
    -- Latest consensus <= trade date for our tickers only
    SELECT *
    FROM (
        SELECT
            b.permno,
            b."date",
            n.ticker,
            s.statpers,
            s.measure,
            s.fiscalp,
            s.fpi,
            s.estflag,
            s.curcode,
            ROW_NUMBER() OVER (
                PARTITION BY b.permno, b."date"
                ORDER BY s.statpers DESC
            ) AS rn
        FROM base b
        JOIN names_filt n
          ON b.permno = n.permno
         AND b."date" BETWEEN n.namedt AND COALESCE(n.nameenddt, DATE '9999-12-31')
        JOIN ibes_stats s
          ON n.ticker = s.ticker
         AND s.statpers <= b."date"
    ) x
    WHERE rn = 1
),
ibes_act_latest AS (
    -- Latest actuals announcement <= trade date for our tickers only
    SELECT *
    FROM (
        SELECT
            b.permno,
            b."date",
            n.ticker,
            a.anndats,
            a.anntims,
            a.pends,
            a.act_measure,
            a.pdicity,
            ROW_NUMBER() OVER (
                PARTITION BY b.permno, b."date"
                ORDER BY a.anndats DESC, a.anntims DESC
            ) AS rn
        FROM base b
        JOIN names_filt n
          ON b.permno = n.permno
         AND b."date" BETWEEN n.namedt AND COALESCE(n.nameenddt, DATE '9999-12-31')
        JOIN ibes_act a
          ON n.ticker = a.ticker
         AND a.anndats <= b."date"
    ) y
    WHERE rn = 1
)
SELECT
    b.permno,
    b."date",
    b.ret         AS crsp_ret,
    b.prc         AS crsp_prc,
    b.cfacpr      AS crsp_cfacpr, -- cumulative factor to adjust prices for splits and dividends
    n.ticker,
    l.gvkey,
    f.datadate    AS comp_datadate,

    f.shs_traded_q,
    f.shs_out_q,
    f.mkt_value_q,
    f.div_ps_exdate_q,
    f.adj_factor_ex,
    f.sales_q,
    f.revenue_q,
    f.cogs_q,
    f.sga_q,
    f.rd_exp_q,
    f.ebitda_q,
    f.ebit_q,
    f.int_exp_q,
    f.net_income_q,
    f.eps_basic_excl_xo_q,
    f.assets_total_q,
    f.liab_total_q,
    f.common_equity_q,
    f.sh_equity_total_q,
    f.cur_assets_q,
    f.cur_liab_q,
    f.cash_sti_q,
    f.inventory_q,
    f.receivables_q,
    f.ppe_net_q,
    f.working_cap_q,
    f.retained_earn_q,
    f.debt_lt_q,
    f.debt_curr_q,
    f.notes_pay_q,
    f.shs_repurchased_q,
    f.avg_repurchase_px_q,

    fac.mktrf     AS ff_mkt_rf,
    fac.smb       AS ff_smb,
    fac.hml       AS ff_hml,
    fac.umd       AS ff_mom,
    fac.rf        AS ff_rf,
    ic.statpers   AS ibes_statpers,
    ic.measure    AS ibes_measure,
    ic.fiscalp    AS ibes_fiscalp,
    ic.fpi        AS ibes_fpi,
    ic.estflag    AS ibes_estflag,
    ic.curcode    AS ibes_curcode,
    ia.anndats    AS ibes_anndats,
    ia.anntims    AS ibes_anntims,
    ia.pends      AS ibes_pends,
    ia.act_measure,
    ia.pdicity,
    l.match_type
FROM base b
LEFT JOIN names_filt n
  ON b.permno = n.permno
 AND b."date" BETWEEN n.namedt AND COALESCE(n.nameenddt, DATE '9999-12-31')
LEFT JOIN links l
  ON b.permno = l.permno
 AND b."date" = l."date"
LEFT JOIN comp_fund f
  ON l.gvkey = f.gvkey
 AND f.datadate <= b."date"
LEFT JOIN ff fac
  ON b."date" = fac."date"
LEFT JOIN ibes_cons_latest ic
  ON b.permno = ic.permno AND b."date" = ic."date"
LEFT JOIN ibes_act_latest ia
  ON b.permno = ia.permno AND b."date" = ia."date"
-- The driver injects per-year date bounds here:
--__DATE_FILTER__