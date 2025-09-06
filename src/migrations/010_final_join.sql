WITH base AS (
    SELECT permno, date, ret, vol, prc, shrout FROM dsf
),
crsp_names AS (
    SELECT permno, ticker, ncusip, namedt, nameenddt FROM stocknames
),
raw_links AS (
    SELECT b.permno, b.date, s.gvkey, s.datadate AS snap_dt,
           1 AS priority, 'CUSIP' AS match_type
    FROM base b
    JOIN crsp_names n ON b.permno = n.permno
                     AND b.date BETWEEN n.namedt AND COALESCE(n.nameenddt, DATE '9999-12-31')
    JOIN secm s ON n.ncusip = s.cusip
               AND s.datadate <= b.date
    UNION ALL
    SELECT b.permno, b.date, s.gvkey, s.datadate AS snap_dt,
           2 AS priority, 'TICKER' AS match_type
    FROM base b
    JOIN crsp_names n ON b.permno = n.permno
                     AND b.date BETWEEN n.namedt AND COALESCE(n.nameenddt, DATE '9999-12-31')
    JOIN secm s ON n.ticker = s.tic
               AND s.datadate <= b.date
),
links AS (
    SELECT * FROM (
        SELECT rl.*, ROW_NUMBER() OVER (
            PARTITION BY rl.permno, rl.date
            ORDER BY rl.priority ASC, rl.snap_dt DESC, rl.gvkey
        ) AS rn
        FROM raw_links rl
    ) z WHERE rn = 1
),
comp_fund AS (
    SELECT gvkey, datadate, at, lt, sale, ni FROM fundq
),
ibes_cons_latest AS (
    SELECT * FROM (
        SELECT b.permno, b.date, n.ticker, s.statpers,
               s.measure, s.fiscalp, s.fpi, s.estflag, s.curcode,
               ROW_NUMBER() OVER (
                   PARTITION BY b.permno, b.date
                   ORDER BY s.statpers DESC
               ) AS rn
        FROM base b
        JOIN crsp_names n ON b.permno = n.permno
                         AND b.date BETWEEN n.namedt AND COALESCE(n.nameenddt, DATE '9999-12-31')
        JOIN ibes_stats s ON n.ticker = s.ticker
                         AND s.statpers <= b.date
    ) x WHERE rn = 1
),
ibes_act_latest AS (
    SELECT * FROM (
        SELECT b.permno, b.date, n.ticker, a.anndats, a.anntims, a.pends,
               a.act_measure, a.pdicity,
               ROW_NUMBER() OVER (
                   PARTITION BY b.permno, b.date
                   ORDER BY a.anndats DESC, a.anntims DESC
               ) AS rn
        FROM base b
        JOIN crsp_names n ON b.permno = n.permno
                         AND b.date BETWEEN n.namedt AND COALESCE(n.nameenddt, DATE '9999-12-31')
        JOIN ibes_act a ON n.ticker = a.ticker
                       AND a.anndats <= b.date
    ) y WHERE rn = 1
)
SELECT COUNT(*) AS row_count
FROM base b
LEFT JOIN crsp_names n ON b.permno = n.permno
                      AND b.date BETWEEN n.namedt AND COALESCE(n.nameenddt, DATE '9999-12-31')
LEFT JOIN links l ON b.permno = l.permno AND b.date = l.date
LEFT JOIN comp_fund f ON l.gvkey = f.gvkey AND f.datadate <= b.date
LEFT JOIN ff fac ON b.date = fac.date
LEFT JOIN ibes_cons_latest ic ON b.permno = ic.permno AND b.date = ic.date
LEFT JOIN ibes_act_latest ia ON b.permno = ia.permno AND b.date = ia.date;