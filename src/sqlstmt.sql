WITH base AS (
    SELECT
        dsf.permno,
        dsf.date,
        dsf.ret,
        dsf.vol,
        dsf.prc,
        dsf.shrout
    FROM crsp.dsf dsf
),

crsp_names AS (
    SELECT
        n.permno,
        n.ticker,
        n.ncusip,
        n.namedt,
        n.nameenddt
    FROM crsp.stocknames n
),

/* CRSP -> Compustat GVKEY via COMP.SECM snapshots:
   pick latest SEC M row with DATADATE <= trade date (prefer CUSIP over TICKER) */
raw_links AS (
    -- Priority 1: CUSIP snapshot match
    SELECT
        b.permno,
        b.date,
        s.gvkey,
        s.datadate AS snap_dt,
        1 AS priority,
        'CUSIP' AS match_type
    FROM base b
    JOIN crsp_names n
      ON b.permno = n.permno
     AND b.date BETWEEN n.namedt AND COALESCE(n.nameenddt, DATE '9999-12-31')
    JOIN comp.secm s
      ON n.ncusip = s.cusip
     AND s.datadate <= b.date

    UNION ALL

    -- Priority 2: TICKER snapshot match
    SELECT
        b.permno,
        b.date,
        s.gvkey,
        s.datadate AS snap_dt,
        2 AS priority,
        'TICKER' AS match_type
    FROM base b
    JOIN crsp_names n
      ON b.permno = n.permno
     AND b.date BETWEEN n.namedt AND COALESCE(n.nameenddt, DATE '9999-12-31')
    JOIN comp.secm s
      ON n.ticker = s.tic
     AND s.datadate <= b.date
),

links AS (
    SELECT *
    FROM (
        SELECT
            rl.*,
            ROW_NUMBER() OVER (
                PARTITION BY rl.permno, rl.date
                ORDER BY rl.priority ASC, rl.snap_dt DESC, rl.gvkey
            ) AS rn
        FROM raw_links rl
    ) z
    WHERE rn = 1
),

comp_fund AS (
    SELECT
        q.gvkey,
        q.datadate,
        q.atq   AS at,
        q.ltq   AS lt,
        q.saleq AS sale,
        q.niq   AS ni
    FROM comp.fundq q
),

ibes_cons_latest AS (
    SELECT *
    FROM (
        SELECT
            b.permno,
            b.date,
            n.ticker,
            s.statpers,
            s.measure,
            s.fiscalp,
            s.fpi,
            s.estflag,
            s.curcode,
            ROW_NUMBER() OVER (
                PARTITION BY b.permno, b.date
                ORDER BY s.statpers DESC
            ) AS rn
        FROM base b
        JOIN crsp_names n
          ON b.permno = n.permno
         AND b.date BETWEEN n.namedt AND COALESCE(n.nameenddt, DATE '9999-12-31')
        JOIN ibes.statsumu_epsus s
          ON n.ticker = s.ticker
         AND s.statpers <= b.date
    ) x
    WHERE rn = 1
),

ibes_act_latest AS (
    SELECT *
    FROM (
        SELECT
            b.permno,
            b.date,
            n.ticker,
            a.anndats,
            a.anntims,
            a.pends,
            a.measure AS act_measure,
            a.pdicity,
            ROW_NUMBER() OVER (
                PARTITION BY b.permno, b.date
                ORDER BY a.anndats DESC, a.anntims DESC
            ) AS rn
        FROM base b
        JOIN crsp_names n
          ON b.permno = n.permno
         AND b.date BETWEEN n.namedt AND COALESCE(n.nameenddt, DATE '9999-12-31')
        JOIN ibes.actu_epsus a
          ON n.ticker = a.ticker
         AND a.anndats <= b.date
    ) y
    WHERE rn = 1
)

SELECT
    b.permno,
    b.date,
    n.ticker,
    l.gvkey,
    f.datadate,
    f.at,
    f.lt,
    f.sale,
    f.ni,
    fac.mktrf AS mkt_rf,
    fac.smb,
    fac.hml,
    fac.umd AS mom,
    fac.rf,
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
LEFT JOIN crsp_names n
  ON b.permno = n.permno
 AND b.date BETWEEN n.namedt AND COALESCE(n.nameenddt, DATE '9999-12-31')
LEFT JOIN links l
  ON b.permno = l.permno
 AND b.date   = l.date
LEFT JOIN comp_fund f
  ON l.gvkey = f.gvkey
 AND f.datadate <= b.date
LEFT JOIN ff.factors_daily fac
  ON b.date = fac.date
LEFT JOIN ibes_cons_latest ic
  ON b.permno = ic.permno AND b.date = ic.date
LEFT JOIN ibes_act_latest ia
  ON b.permno = ia.permno AND b.date = ia.date
LIMIT 200;