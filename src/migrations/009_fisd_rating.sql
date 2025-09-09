-- Ratings data for Corporate Bonds
-- Source: FISD (Fixed Income Securities Database) via WRDS
SELECT
  issue_id,
  rating_date AS "date",
  rating_type,
  rating,
  rating_status,
  reason,
  rating_status_date,
  investment_grade
FROM fisd.rating
WHERE rating_date >= %(start)s::date
  AND rating_date <  %(end)s::date;