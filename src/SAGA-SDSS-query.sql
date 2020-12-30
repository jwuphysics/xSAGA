SELECT
    objID, 
    ra, 
    dec, 
    dered_g AS g0,
    dered_r AS r0,
    petror50_r AS R_eff,
FROM Galaxy 
WHERE dered_r <= 21.0
    and (flags_r & 0x0000000010000000) != 0 -- detected in BINNED1
    and (flags_r & 0x0000000000040000)  = 0 -- no SATURATED
    and (flags_r & 0x0000010000000000)  = 0 -- no BAD_COUNTS_ERROR
    and (((ABS(modelMagErr_g) < 0.5) and (ABS(modelMagErr_r) < 0.5))
      or ((ABS(modelMagErr_g) < 0.5) and (ABS(modelMagErr_i) < 0.5)) 
      or ((ABS(modelMagErr_r) < 0.5) and (ABS(modelMagErr_i) < 0.5)) )
    and ((dered_g - dered_r) <= 10 
     and (dered_r - dered_i) <= 10 
     and (dered_u - dered_r) <= 10 )
    and ((dered_g - dered_r) - 
         SQRT(POWER(modelMagErr_g, 2) + POWER(modelMagErr_r, 2)) +
         0.06 * (dered_r - 14) < 0.9)                   -- color cut
    and (dered_r + 2.5 * LOG10(6.28318531 * POWER(petror50_r, 2)) +
         SQRT(POWER(modelMagErr_r, 2) + 2.17147 * ABS(petroR50Err_r / petroR50_r)) - 
         0.7 * (dered_r - 14) > 18.5                    -- surface brightness cut 
    )