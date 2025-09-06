SELECT n.permno, n.ticker, n.ncusip, n.namedt, n.nameenddt
FROM crsp.stocknames n
WHERE EXISTS (
    SELECT 1
    FROM crsp.dsf d
    WHERE d.permno = n.permno
      AND d.date BETWEEN %(start)s AND %(end)s
);