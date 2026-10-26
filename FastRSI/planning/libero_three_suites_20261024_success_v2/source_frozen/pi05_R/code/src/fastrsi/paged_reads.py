"""Bounded complete reads through the native line-numbered read_file protocol."""
import re

class IncompleteRead(ValueError):
    pass

async def read_complete(execute,path,*,on_page,max_pages=32,max_chars=4_096_000,missing_ok=False):
    """Follow verified native continuation markers; never write or infer missing text."""
    offset=1;known_total=None;lines=[];size=0
    for page in range(max_pages):
        params={'path':path}
        if offset!=1:params['offset']=offset
        result=await execute('read_file',params)
        if result.startswith('Error'):
            if page==0 and missing_ok and result.startswith('Error: File not found:'):
                on_page({'tool':'read_file',**params,'observed_missing':True});return None
            raise IncompleteRead(result)
        if result.startswith('(Empty file:') and offset==1:
            on_page({'tool':'read_file',**params,'lines':0});return ''
        footer=re.search(r'\n\n\((Showing lines (\d+)-(\d+) of (\d+)\. Use offset=(\d+) to continue\.|End of file — (\d+) lines total)\)\Z',result)
        if footer is None:raise IncompleteRead('Missing complete native pagination marker')
        numbered=result[:footer.start()].splitlines();chunk=[]
        for index,line in enumerate(numbered,offset):
            match=re.fullmatch(r'(\d+)\| (.*)',line)
            if not match or int(match[1])!=index:raise IncompleteRead('Gapped or repeated native line numbers')
            chunk.append(match[2])
        if not chunk:raise IncompleteRead('Read did not advance')
        end=offset+len(chunk)-1
        if footer[6] is not None:
            total=int(footer[6]);done=True
            if end!=total:raise IncompleteRead('EOF marker disagrees with returned lines')
        else:
            start,reported_end,total,next_offset=map(int,footer.group(2,3,4,5));done=False
            if (start,reported_end,next_offset)!=(offset,end,end+1) or end>=total:
                raise IncompleteRead('Invalid native continuation range')
        if known_total is not None and total!=known_total:raise IncompleteRead('File length changed between pages')
        known_total=total;size+=sum(len(x)+1 for x in chunk)
        if size>max_chars:raise IncompleteRead('Complete read exceeds character budget')
        lines.extend(chunk);on_page({'tool':'read_file',**params,'lines':len(chunk),'total_lines':total})
        if done:return '\n'.join(lines)
        offset=end+1
    raise IncompleteRead('Complete read exceeds page budget')
