# Findings Report

## 1. Dataset Description

The dataset contains approximately 2,000 student records with multiple academic and behavioral variables, including GPA, study hours, attendance, department, and scholarship type.

Both numeric and categorical features are present. Some data quality issues were identified:

* `commute_minutes` has approximately 10% missing values, which were imputed using the median.
* `study_hours_weekly` has approximately 5% missing values, which were handled by dropping those rows.

Overall, the dataset is clean and suitable for analysis after preprocessing.



## 2. Key Distribution Findings

The GPA distribution is slightly left-skewed, indicating that most students cluster between 2.5 and 3.5 (see `output/gpa_distribution.png`).

The distribution of weekly study hours shows moderate variation across students, suggesting differences in study habits (see `output/study_hours_weekly_distribution.png`).

Attendance percentage is generally high but varies across individuals, which may impact academic outcomes (see `output/attendance_pct_distribution.png`).

The boxplot comparing GPA across departments reveals noticeable differences in median GPA between departments, suggesting variation in academic performance or grading standards (see `output/gpa_by_department_boxplot.png`).

The distribution of scholarship types shows that some categories are more common than others, indicating uneven allocation across students (see `output/scholarship_distribution.png`).



## 3. Notable Correlations

The correlation heatmap highlights relationships between numeric variables (see `output/correlation_heatmap.png`).

The strongest positive correlations were observed between:

* `study_hours_weekly` and `gpa`
* `attendance_pct` and `gpa`

Scatter plots confirm these relationships:

* Study hours vs GPA shows a clear upward trend (see `output/top_correlation_pair_1_gpa_vs_study_hours_weekly.png`)
* Attendance vs GPA also shows a positive association (see `output/top_correlation_pair_2_attendance_pct_vs_gpa.png`)

These findings suggest that increased academic engagement (more study time and higher attendance) is associated with better academic performance.

However, it is important to note that correlation does not imply causation. Other factors such as motivation, prior ability, or external conditions may influence these relationships.



## 4. Hypothesis Test Results

### Hypothesis 1:

Students with internships have a higher GPA than students without internships.

* Mean GPA (Yes): 2.98
* Mean GPA (No): 2.70
* t-statistic: 14.23
* p-value: 0.0000
* Cohen’s d: 0.69

Interpretation:
There is a statistically significant difference in GPA between students with and without internships (p < 0.05). The effect size (Cohen’s d ≈ 0.69) indicates a moderate practical impact, meaning the difference is not only statistically significant but also meaningful in practice.



### Hypothesis 2:

Scholarship status is associated with department.

* Chi-square statistic: 13.95
* p-value: 0.3040
* Degrees of freedom: 12

Interpretation:
There is no statistically significant association between scholarship type and department (p > 0.05). This suggests that scholarship distribution is relatively independent of department.


## 5. Actionable Recommendations

1. Promote structured study programs
   Since study hours are positively correlated with GPA, the university should encourage structured study schedules and provide academic support programs to improve student performance.

2. Improve attendance engagement strategies
   Given the strong relationship between attendance and GPA, implementing attendance incentives or more engaging teaching methods could improve academic outcomes.

3. Expand internship opportunities
   Students with internships tend to achieve higher GPAs. The university should expand internship programs and integrate them more effectively into academic pathways.



## Conclusion

The analysis shows that academic engagement factors such as study time and attendance are strongly associated with student performance. Internship participation also appears to have a meaningful impact on GPA, while scholarship type does not significantly vary by department. These insights can help guide institutional strategies to improve student success.
