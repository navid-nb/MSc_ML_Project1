SELECT gvkey, tic, cusip, datadate
FROM comp.secm s
WHERE datadate <= %(end)s
  AND (
    tic IN (
      SELECT DISTINCT n.ticker
      FROM crsp.stocknames n
      JOIN crsp.dsf d ON d.permno = n.permno
      WHERE d.date BETWEEN %(start)s AND %(end)s
    )
    OR cusip IN (
      SELECT DISTINCT n.ncusip
      FROM crsp.stocknames n
      JOIN crsp.dsf d ON d.permno = n.permno
      WHERE d.date BETWEEN %(start)s AND %(end)s
    )
  );