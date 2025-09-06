-- CRSP name history (trim to relevant window to shrink payload)
SELECT
  permno,
  ticker,
  ncusip,
  namedt,
  nameenddt
FROM crsp.stocknames
WHERE COALESCE(nameenddt, DATE '9999-12-31') >= %(start)s::date
  AND namedt <= %(end)s::date;
