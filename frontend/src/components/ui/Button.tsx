import { ButtonHTMLAttributes, forwardRef } from "react";
import { cx } from "@/lib/utils";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md" | "lg";

const variantClasses: Record<Variant, string> = {
  primary:
    "bg-gradient-to-r from-accent-cyan to-accent-teal text-base-950 hover:brightness-110 shadow-glow font-semibold",
  secondary: "bg-white/8 text-slate-100 hover:bg-white/12 border border-white/10",
  ghost: "bg-transparent text-slate-300 hover:bg-white/6 hover:text-white",
  danger: "bg-accent-rose/15 text-accent-rose border border-accent-rose/30 hover:bg-accent-rose/25",
};

const sizeClasses: Record<Size, string> = {
  sm: "px-3 py-1.5 text-xs",
  md: "px-4 py-2.5 text-sm",
  lg: "px-6 py-3 text-base",
};

export const Button = forwardRef<
  HTMLButtonElement,
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: Size }
>(({ variant = "primary", size = "md", className, children, disabled, ...props }, ref) => {
  return (
    <button
      ref={ref}
      disabled={disabled}
      className={cx(
        "inline-flex items-center justify-center gap-2 rounded-xl transition-all duration-200",
        "focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-cyan/50",
        disabled && "opacity-50 cursor-not-allowed",
        variantClasses[variant],
        sizeClasses[size],
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
});
Button.displayName = "Button";
