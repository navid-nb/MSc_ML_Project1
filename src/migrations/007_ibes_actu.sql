SELECT ticker, anndats, anntims, pends, measure AS act_measure, pdicity
FROM ibes.actu_epsus a
WHERE anndats <= %(end)s
  AND ticker IN (
    SELECT DISTINCT n.ticker
    FROM crsp.stocknames n
    JOIN crsp.dsf d ON d.permno = n.permno
    WHERE d.date BETWEEN %(start)s AND %(end)s
  );