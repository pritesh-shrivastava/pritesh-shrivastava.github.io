(define (identityM n)
    (letrec 
        (
          ;; N -> [List-of [List-of N]]
          (generate-matrix (lambda (row)
            (cond
             ((= row 0) '())
             (else (cons (generate-row row n)
                         (generate-matrix (- row 1)))))))

          ;; N N -> [List-of N]
          (generate-row (lambda (row col)
            ;; col goes from column n to 0
            (cond
             ((= col 0) '())
             (else (cons (if (= row col) 1 0)
                         (generate-row row (- col 1)))))))
        )
        (generate-matrix n)
    )
)
      
(identityM 3)