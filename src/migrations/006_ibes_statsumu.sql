SELECT ticker, statpers, measure, fiscalp, fpi, estflag, curcode
FROM ibes.statsumu_epsus s
WHERE statpers <= %(end)s
  AND ticker IN (
    SELECT DISTINCT n.ticker
    FROM crsp.stocknames n
    JOIN crsp.dsf d ON d.permno = n.permno
    WHERE d.date BETWEEN %(start)s AND %(end)s
  );