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
        dsf.shrout
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
        q.gvkey,
        q.datadate,
        q."at",
        q."lt",
        q."sale",
        q."ni"
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
    n.ticker,
    l.gvkey,
    f.datadate    AS comp_datadate,
    f."at"        AS comp_at,
    f."lt"        AS comp_lt,
    f."sale"      AS comp_sale,
    f."ni"        AS comp_ni,
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