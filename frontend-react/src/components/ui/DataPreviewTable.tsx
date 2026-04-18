import { cn } from '@/lib/utils'

interface Props {
  columns: string[]
  rows: Record<string, unknown>[]
}

export function DataPreviewTable({ columns, rows }: Props) {
  if (!columns.length || !rows.length) return null

  return (
    <div className="overflow-x-auto rounded-lg border border-border">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-border bg-accent">
            {columns.map((col) => (
              <th
                key={col}
                className="text-left px-3 py-2 font-medium text-muted-foreground whitespace-nowrap"
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={i}
              className={cn(
                'border-b border-border last:border-0',
                i % 2 === 0 ? 'bg-card' : 'bg-accent/30'
              )}
            >
              {columns.map((col) => {
                const val = row[col]
                const isNull = val === null || val === undefined || val === ''
                return (
                  <td
                    key={col}
                    className={cn(
                      'px-3 py-2 whitespace-nowrap max-w-[180px] truncate font-mono',
                      isNull ? 'text-slate-600 italic' : 'text-slate-300'
                    )}
                  >
                    {isNull ? 'null' : String(val)}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
