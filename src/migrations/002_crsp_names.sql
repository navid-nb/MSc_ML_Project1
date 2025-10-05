-- CRSP stocknames
SELECT permno    -- CRSP permanent security number
     , ticker    -- ticker symbol
     , ncusip    -- issue-level CUSIP identifier
     , namedt    -- start date when this name/ticker/CUSIP record is valid
     , nameenddt -- end date when this record is valid
FROM crsp.stocknames
WHERE COALESCE(nameenddt, DATE '9999-12-31') >= %(start)s::date  -- security still active after start date
  AND namedt <= %(end)s::date -- security became active before end date
  AND exchcd = 3; -- nasdaq