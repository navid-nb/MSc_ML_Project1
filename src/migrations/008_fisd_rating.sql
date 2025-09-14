-- Fixed Income Securities Database (fisd) ratings data for Corporate Bonds
WITH base AS (SELECT issue_id,           -- Unique bond issue identifier
                     rating_date,        -- Date of the rating action
                     rating_type,        -- Agency / rating family (S&P, Moody’s, Fitch, ...)
                     rating,             -- Assigned alphanumeric credit rating (AAA, ..., D)
                     rating_status,      -- Status: New, Upgrade, Downgrade, Affirmation, Watch, Withdrawn
                     reason,             -- Reason code for rating action
                     rating_status_date, -- Effective date of status change
                     investment_grade,   -- Y/N flag for investment grade classification

                     -- numerical rating
                     CASE
                         WHEN rating IN ('AAA', 'Aaa') THEN 22
                         WHEN rating IN ('AA+', 'Aa1') THEN 21
                         WHEN rating IN ('AA', 'Aa2') THEN 20
                         WHEN rating IN ('AA-', 'Aa3') THEN 19
                         WHEN rating IN ('A+', 'A1') THEN 18
                         WHEN rating IN ('A', 'A2') THEN 17
                         WHEN rating IN ('A-', 'A3') THEN 16
                         WHEN rating IN ('BBB+', 'Baa1') THEN 15
                         WHEN rating IN ('BBB', 'Baa2') THEN 14
                         WHEN rating IN ('BBB-', 'Baa3') THEN 13
                         WHEN rating IN ('BB+', 'Ba1') THEN 12
                         WHEN rating IN ('BB', 'Ba2') THEN 11
                         WHEN rating IN ('BB-', 'Ba3') THEN 10
                         WHEN rating IN ('B+', 'B1') THEN 9
                         WHEN rating IN ('B', 'B2') THEN 8
                         WHEN rating IN ('B-', 'B3') THEN 7
                         WHEN rating = 'CCC+' THEN 6
                         WHEN rating = 'CCC' THEN 5
                         WHEN rating = 'CCC-' THEN 4
                         WHEN rating = 'CC' THEN 3
                         WHEN rating = 'C' THEN 2
                         WHEN rating = 'D' THEN 1
                         ELSE 0 -- Fallback for non rated or unusual codes
                         END AS rating_num
              FROM fisd.fisd_ratings
              WHERE rating_date >= %(start)s::date
    AND rating_date <  %(end)s::date
)
SELECT
    -- Identifiers
    issue_id,                                                                              -- Bond issue ID
    rating_date                                                        AS rating_dt,       -- Event date
    rating_type,                                                                           -- Agency / rating family
    rating,                                                                                -- Alphanumeric rating
    rating_status,                                                                         -- Provider’s status label
    reason,                                                                                -- Reason code
    rating_status_date,                                                                    -- Effective date of status
    investment_grade,                                                                      -- Y/N flag (can be cast to 1/0)
    CASE
        WHEN investment_grade = 'Y' THEN 1
        ELSE 0
        END                                                            AS investment_grade_flag,
    -- Previous state
    LAG(rating)
        OVER (PARTITION BY issue_id, rating_type ORDER BY rating_date) AS prev_rating,     -- Previous alphanumeric rating
    LAG(rating_num)
        OVER (PARTITION BY issue_id, rating_type ORDER BY rating_date) AS prev_rating_num, -- Previous numeric rating

    -- Categorical change label
    CASE
        WHEN LAG(rating_num) OVER (PARTITION BY issue_id, rating_type ORDER BY rating_date) IS NULL
            THEN 'NEW' -- First observation for this issue/agency
        WHEN rating_num > LAG(rating_num) OVER (PARTITION BY issue_id, rating_type ORDER BY rating_date)
            THEN 'UPGRADE' -- Numeric rating improved
        WHEN rating_num < LAG(rating_num) OVER (PARTITION BY issue_id, rating_type ORDER BY rating_date)
            THEN 'DOWNGRADE' -- Numeric rating worsened
        ELSE 'UNCHANGED' -- No change
        END                                                            AS rating_change_label,

    -- Ordinal numeric rating
    rating_num,

    -- Directional intensity
    rating_num
        - LAG(rating_num) OVER (PARTITION BY issue_id, rating_type ORDER BY rating_date)
                                                                       AS rating_diff,     -- Positive = upgrade size, negative = downgrade size, 0 = unchanged, NULL = new

    -- Signed direction only (-1, 0, +1)
    CASE
        WHEN LAG(rating_num) OVER (PARTITION BY issue_id, rating_type ORDER BY rating_date) IS NULL THEN NULL
        WHEN rating_num > LAG(rating_num) OVER (PARTITION BY issue_id, rating_type ORDER BY rating_date) THEN 1
        WHEN rating_num < LAG(rating_num) OVER (PARTITION BY issue_id, rating_type ORDER BY rating_date) THEN -1
        ELSE 0
        END                                                            AS rating_dir,      -- Simple indicator: +1 upgrade, -1 downgrade, 0 unchanged

    -- Magnitude only (absolute value of move)
    CASE
        WHEN LAG(rating_num) OVER (PARTITION BY issue_id, rating_type ORDER BY rating_date) IS NULL THEN NULL
        ELSE ABS(rating_num - LAG(rating_num) OVER (PARTITION BY issue_id, rating_type ORDER BY rating_date))
        END                                                            AS rating_mag       -- Size of change in notches (e.g., 2 = moved two notches)
FROM base
ORDER BY issue_id, rating_dt;