import type { ReactNode } from "react";

/** Consistent instrument chrome for every stage scene. */
export function Stage({
  label,
  labelValue,
  foot,
  children,
  aria,
}: {
  label: string;
  labelValue?: ReactNode;
  foot?: ReactNode;
  children?: ReactNode;
  aria?: string;
}) {
  return (
    <>
      <span className="stage-label">
        {label}
        {labelValue ? <> <b>{labelValue}</b></> : null}
      </span>
      <span className="stage-corner stage-corner--tl" aria-hidden="true" />
      <span className="stage-corner stage-corner--tr" aria-hidden="true" />
      <span className="stage-corner stage-corner--bl" aria-hidden="true" />
      <span className="stage-corner stage-corner--br" aria-hidden="true" />
      <div className="stage-body" role="img" aria-label={aria ?? label}>
        {children}
      </div>
      {foot && <div className="stage-foot">{foot}</div>}
    </>
  );
}
