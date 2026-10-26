"""Allow explicitly absent non-executable lesson metadata; retain strict decisions."""
from final_contract import validate_verdict as strict_validate

def validate_verdict(value):
 if not isinstance(value,dict):return strict_validate(value)
 result=dict(value)
 lesson=result.get('lesson')
 if lesson is None or (isinstance(lesson,str) and not lesson.strip()):
  result['lesson']='NOT_PROVIDED_BY_VERIFIER'
  result['lesson_missing']=True
  result['metadata_normalization']='optional_lesson_v1'
 return strict_validate(result)
